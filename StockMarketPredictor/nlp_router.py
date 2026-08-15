"""
Phase 1: NLP Metadata Router
=============================
Stock Market Predictor & Financial Intelligence Agent

Loads a pre-trained Keras Bag-of-Words classifier (21K financial tweets,
20 imbalanced categories, 0.83 Weighted F1) along with its fitted
vectorizer and label encoder, then exposes `tag_financial_news(text)`
which cleans, tokenizes, and lemmatizes a raw headline/tweet and returns
the predicted topic label used to tag downstream data (see Phase 2's
ChromaDB metadata filter).

Expected artifacts on disk (produced by the original training run):
    artifacts/
        financial_topic_classifier.keras  # trained Keras model
        vectorizer.pkl                    # fitted sklearn CountVectorizer
                                           # (or TfidfVectorizer) - BoW
        label_encoder.pkl                 # OPTIONAL: fitted sklearn
                                           # LabelEncoder mapping class idx
                                           # -> topic str. If absent, the
                                           # router falls back to numeric
                                           # class-index labels ("class_7")
                                           # and logs a loud warning - see
                                           # _load_label_encoder below.

Filenames/paths are overridable via environment variables so this matches
whatever your training run actually produced without editing code:
    ARTIFACTS_DIR             (default: "artifacts")
    KERAS_MODEL_FILENAME      (default: "financial_topic_classifier.keras")
    VECTORIZER_FILENAME       (default: "vectorizer.pkl")
    LABEL_ENCODER_FILENAME    (default: "label_encoder.pkl")

If your vectorizer was instead persisted as a Keras `TextVectorization`
layer, see the `_load_vectorizer` fallback branch below.

MOCK MODE: if the model or vectorizer file is missing at load time (e.g.
you're standing up the rest of the pipeline before training is done), this
module does NOT crash. It logs a prominent warning, sets the module-level
`MOCK_MODE` flag to True, and `tag_financial_news` starts returning a
randomly chosen topic from `MOCK_TOPICS` instead of a real prediction.
This keeps Phase 2 (indexing), Phase 4 (the agent), and Phase 5 (the API
container) fully runnable end-to-end with placeholder labels while you
finish training the real classifier - swap in the real artifacts and
restart to get real predictions again.
"""

from __future__ import annotations

import logging
import os
import pickle
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import nltk
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / os.getenv(
    "KERAS_MODEL_FILENAME", "financial_topic_classifier.keras"
)
VECTORIZER_PATH = ARTIFACTS_DIR / os.getenv("VECTORIZER_FILENAME", "vectorizer.pkl")
LABEL_ENCODER_PATH = ARTIFACTS_DIR / os.getenv(
    "LABEL_ENCODER_FILENAME", "label_encoder.pkl"
)

# Confidence threshold below which we fall back to a generic label rather
# than a low-confidence guess. Tune against your validation set.
MIN_CONFIDENCE = 0.15
FALLBACK_LABEL = "General/Uncategorized"

# --------------------------------------------------------------------------- #
# Mock Mode: set by _load_artifacts() the first time it runs (e.g. via
# warm_up() at API startup). True whenever the real model/vectorizer
# artifacts weren't found on disk. Importable elsewhere (e.g.
# `nlp_router.MOCK_MODE`) to surface a degraded-mode banner in /health or
# logs - but its value is only accurate AFTER _load_artifacts() has run
# once; before that it defaults to False.
# --------------------------------------------------------------------------- #

MOCK_MODE = False

MOCK_TOPICS = ["Earnings", "Macroeconomics", "M&A", "Tech", "Regulatory"]

# --------------------------------------------------------------------------- #
# NLTK setup (idempotent - only downloads if not already present)
# --------------------------------------------------------------------------- #


def _ensure_nltk_resources() -> None:
    """Download required NLTK corpora on first run only."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "corpora/stopwords": "stopwords",
    }
    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading missing NLTK resource: %s", download_name)
            nltk.download(download_name, quiet=True)


_ensure_nltk_resources()

from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))

# Financial text carries signal in tokens a generic stopword list would
# strip or that generic cleanup would destroy (tickers, %, $). Keep the
# stopword removal conservative and preserve cashtags/percentages upstream
# of tokenization instead of deleting them.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_HASH_RE = re.compile(r"#(\w+)")  # keep the word, drop the hash
_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,6})\b")  # e.g. $AAPL -> TICKER_AAPL
_NON_ALPHANUM_RE = re.compile(r"[^a-zA-Z0-9_\s%.]")
_MULTI_SPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def clean_text(raw_text: str) -> str:
    """Regex-based cleanup of a raw headline/tweet prior to tokenization.

    - Lowercases
    - Strips URLs and @mentions
    - Normalizes cashtags ($AAPL -> ticker_aapl) so the BoW vocabulary
      treats them as first-class financial tokens instead of dropping them
    - Strips hashtags' leading '#' but keeps the word
    - Removes remaining punctuation/symbols except '%' and '.' (decimals)
    """
    text = raw_text.strip().lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _CASHTAG_RE.sub(r"ticker_\1", text)
    text = _HASHTAG_HASH_RE.sub(r"\1", text)
    text = _NON_ALPHANUM_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize_and_lemmatize(cleaned_text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize cleaned text and lemmatize each token (verb + noun pass)."""
    tokens = word_tokenize(cleaned_text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    lemmatized = []
    for tok in tokens:
        # Lemmatize as verb first (handles e.g. "surged" -> "surge"),
        # then as noun for anything the verb pass left unchanged
        # (handles e.g. "earnings" -> "earning").
        lemma = _LEMMATIZER.lemmatize(tok, pos="v")
        lemma = _LEMMATIZER.lemmatize(lemma, pos="n")
        lemmatized.append(lemma)

    return lemmatized


def preprocess(raw_text: str) -> str:
    """Full preprocessing pipeline -> a whitespace-joined string ready for
    the fitted BoW vectorizer's `.transform()`.
    """
    cleaned = clean_text(raw_text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# Artifact loading (lazy + cached so this module is cheap to import and
# tests can mock `_load_artifacts` without paying model-load cost)
# --------------------------------------------------------------------------- #


@dataclass
class RouterArtifacts:
    model: "tf.keras.Model"  # noqa: F821
    vectorizer: object
    label_encoder: object


def _load_vectorizer(path: Path):
    """Load a fitted sklearn (Count/Tfidf)Vectorizer from pickle.

    If your pipeline instead persisted a Keras `TextVectorization` layer
    (e.g. saved inside a `tf.keras.Sequential` adapter model), swap this
    for `tf.keras.models.load_model(path)` and call `.predict` /
    `layer(...)` instead of `.transform(...)` in `_vectorize`.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_label_encoder(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


class _FallbackLabelEncoder:
    """Used only when LABEL_ENCODER_PATH doesn't exist. Keeps the same
    `.inverse_transform(...)` interface as sklearn's LabelEncoder so
    `tag_financial_news_detailed` doesn't need to branch on which one it
    got - but produces "class_<idx>" placeholders instead of real topic
    names, since we have no mapping without the real artifact. This lets
    the service boot and serve predictions rather than hard-failing, but
    Phase 2's metadata filtering will only ever see these placeholder
    labels until a real label_encoder.pkl is supplied.
    """

    def inverse_transform(self, indices):
        return [f"class_{int(i)}" for i in indices]


@lru_cache(maxsize=1)
def _load_artifacts(
    model_path: Path = MODEL_PATH,
    vectorizer_path: Path = VECTORIZER_PATH,
    label_encoder_path: Path = LABEL_ENCODER_PATH,
) -> Optional[RouterArtifacts]:
    """Load and cache the Keras model + vectorizer + label encoder.

    Cached via lru_cache so repeated calls to `tag_financial_news` (e.g.
    inside an API server handling many requests) do not repay model-load
    cost each time. Call `_load_artifacts.cache_clear()` in tests to force
    a reload with different paths/mocks.

    Returns None (and sets the module-level MOCK_MODE flag to True) if
    the mandatory model or vectorizer file is missing, instead of raising
    - see the module docstring's "MOCK MODE" section. Model + vectorizer
    are both mandatory: there's no way to produce a real prediction with
    only one of them, so either being absent triggers mock mode.
    """
    global MOCK_MODE

    missing = [p for p in (model_path, vectorizer_path) if not p.exists()]
    if missing:
        MOCK_MODE = True
        missing_list = ", ".join(str(p) for p in missing)
        logger.warning(
            "\n" + "=" * 72 + "\n"
            "  MOCK MODE ENABLED - NLP classifier artifacts not found\n"
            "  Missing: %s\n"
            "  tag_financial_news() will return RANDOMLY CHOSEN topics from\n"
            "  %s instead of real predictions.\n"
            "  Place your trained model + vectorizer in '%s' and restart\n"
            "  this service to exit mock mode.\n" + "=" * 72,
            missing_list, MOCK_TOPICS, ARTIFACTS_DIR,
        )
        return None

    MOCK_MODE = False

    import tensorflow as tf  # local import: keep TF off the import path
    # for callers that only need `clean_text`/`preprocess` (e.g. Phase 2).

    logger.info("Loading Keras classifier from %s", model_path)
    model = tf.keras.models.load_model(model_path)

    logger.info("Loading vectorizer from %s", vectorizer_path)
    vectorizer = _load_vectorizer(vectorizer_path)

    # Label encoder is OPTIONAL: fall back to placeholder class names
    # rather than refusing to start, since a missing/renamed label
    # encoder shouldn't take down the whole service.
    if label_encoder_path.exists():
        logger.info("Loading label encoder from %s", label_encoder_path)
        label_encoder = _load_label_encoder(label_encoder_path)
    else:
        logger.warning(
            "No label encoder found at %s - falling back to placeholder "
            "'class_<idx>' topic labels. Predictions will still work, but "
            "Phase 2's topic metadata filter won't match real category "
            "names until a real label_encoder.pkl is supplied.",
            label_encoder_path,
        )
        label_encoder = _FallbackLabelEncoder()

    return RouterArtifacts(model=model, vectorizer=vectorizer, label_encoder=label_encoder)


def _vectorize(vectorizer, processed_text: str) -> np.ndarray:
    """Transform preprocessed text into the BoW feature vector the Keras
    model expects. Handles both sparse (sklearn) and dense outputs.
    """
    features = vectorizer.transform([processed_text])
    if hasattr(features, "toarray"):
        features = features.toarray()
    return np.asarray(features, dtype="float32")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


@dataclass
class TaggedNews:
    text: str
    topic: str
    confidence: float


def tag_financial_news(text: str) -> str:
    """Predict the topic label for a raw financial headline/tweet.

    Parameters
    ----------
    text : str
        Raw, unprocessed headline or tweet text.

    Returns
    -------
    str
        Predicted topic label (one of the trained categories), or
        FALLBACK_LABEL if the model's top prediction confidence is below
        MIN_CONFIDENCE. If the real model artifacts aren't available, this
        is instead a RANDOM label from MOCK_TOPICS - see MOCK MODE in the
        module docstring, and check `nlp_router.MOCK_MODE` if you need to
        know which mode produced a given result.

    Raises
    ------
    ValueError
        If `text` is empty/whitespace-only after cleaning.
    """
    result = tag_financial_news_detailed(text)
    return result.topic


def tag_financial_news_detailed(text: str) -> TaggedNews:
    """Same as `tag_financial_news` but returns the full prediction
    (topic + confidence) for callers that want to log or threshold on it
    (e.g. routing low-confidence items to human review before indexing
    into ChromaDB in Phase 2).
    """
    if not text or not text.strip():
        raise ValueError("tag_financial_news received empty text.")

    artifacts = _load_artifacts()

    if artifacts is None:
        # MOCK MODE: no real classifier to run - hand back a random label
        # so the rest of the pipeline (Phase 2 indexing, Phase 4's agent,
        # Phase 5's API) stays fully exercisable while the real model is
        # still being trained. confidence=0.0 is a placeholder, not a
        # real score - there's nothing to be confident about here.
        topic = random.choice(MOCK_TOPICS)
        return TaggedNews(text=text, topic=topic, confidence=0.0)

    processed = preprocess(text)
    if not processed:
        logger.warning("Text reduced to empty string after preprocessing: %r", text)
        return TaggedNews(text=text, topic=FALLBACK_LABEL, confidence=0.0)

    features = _vectorize(artifacts.vectorizer, processed)
    probabilities = artifacts.model.predict(features, verbose=0)[0]

    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx])

    if confidence < MIN_CONFIDENCE:
        logger.info(
            "Low-confidence prediction (%.3f) for %r; using fallback label.",
            confidence,
            text,
        )
        return TaggedNews(text=text, topic=FALLBACK_LABEL, confidence=confidence)

    topic = artifacts.label_encoder.inverse_transform([top_idx])[0]
    return TaggedNews(text=text, topic=str(topic), confidence=confidence)


def warm_up() -> None:
    """Eagerly load model artifacts (call at API startup, e.g. FastAPI's
    `lifespan`, so the first real request in Phase 5 isn't slowed by the
    model-load cost). Also the earliest point MOCK_MODE becomes accurate -
    call this before checking `nlp_router.MOCK_MODE` elsewhere.
    """
    _load_artifacts()
    if MOCK_MODE:
        logger.warning(
            "NLP Metadata Router warmed up in MOCK MODE - topic labels are "
            "random, not from a trained model."
        )
    else:
        logger.info("NLP Metadata Router warmed up and ready.")


# --------------------------------------------------------------------------- #
# Manual smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    sample_headlines = [
        "$AAPL smashes Q3 earnings estimates, guidance raised for FY24",
        "Fed signals possible rate cuts amid cooling inflation data",
        "Breaking: $TSLA recalls 200k vehicles over autopilot software bug",
    ]

    for headline in sample_headlines:
        tagged = tag_financial_news_detailed(headline)
        print(f"[{tagged.confidence:.2%}] {tagged.topic:25s} <- {headline}")
