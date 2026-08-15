"""
Phase 1 Training Script: Financial Topic Classifier
=====================================================
Stock Market Predictor & Financial Intelligence Agent

Trains the Keras Bag-of-Words classifier that `nlp_router.py` expects to
find in `artifacts/` at startup, using the public "Twitter Financial News
Topic" dataset (zeroshot/twitter-financial-news-topic on Hugging Face):
21,107 tweets across 20 topic categories - a very close match (and quite
possibly the exact original source) for the "21K tweets, 20 categories"
dataset this project was originally scoped around.

WHAT THIS PRODUCES
-------------------
Three artifacts in `artifacts/`, matching nlp_router.py's expected
filenames exactly:
    financial_topic_classifier.keras   # trained Keras model
    vectorizer.pkl                     # fitted sklearn CountVectorizer
    label_encoder.pkl                  # fitted sklearn LabelEncoder

Once these exist, restart the API and nlp_router.py will automatically
exit Mock Mode and start returning real predictions - no code changes
needed, since it was built to detect these files at startup.

WHY THIS REUSES nlp_router.py's OWN PREPROCESSING
----------------------------------------------------
This script imports `clean_text`/`preprocess` directly from nlp_router.py
instead of reimplementing them, so training and inference are
GUARANTEED to preprocess text identically. Training a vectorizer on text
cleaned one way and then serving it with a different cleanup function is
a classic silent train/serve skew bug - importing the real functions
makes that class of bug structurally impossible here, not just
"unlikely because we were careful."

HOW TO RUN
-----------
This needs to run somewhere with real internet access to Hugging Face
(your own machine or EC2 - NOT inside an environment with a restrictive
network proxy). From the project root (same folder as nlp_router.py):

    pip install datasets scikit-learn tensorflow-cpu pandas
    python train_classifier.py

Takes a few minutes on an ordinary CPU. Prints real evaluation metrics
on the held-out validation set at the end - read them, don't assume a
number in advance.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# Import the REAL preprocessing pipeline from nlp_router.py (see module
# docstring above for why this matters) rather than reimplementing it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlp_router import preprocess  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_OUT = ARTIFACTS_DIR / "financial_topic_classifier.keras"
VECTORIZER_OUT = ARTIFACTS_DIR / "vectorizer.pkl"
LABEL_ENCODER_OUT = ARTIFACTS_DIR / "label_encoder.pkl"

# zeroshot/twitter-financial-news-topic's numeric label -> topic name.
# Confirmed against the dataset card on Hugging Face (huggingface.co/
# datasets/zeroshot/twitter-financial-news-topic) - if the upstream
# dataset ever changes this mapping, re-check it there before retraining.
ID2LABEL = {
    0: "Analyst Update",
    1: "Fed | Central Banks",
    2: "Company | Product News",
    3: "Treasuries | Corporate Debt",
    4: "Dividend",
    5: "Earnings",
    6: "Energy | Oil",
    7: "Financials",
    8: "Currencies",
    9: "General News | Opinion",
    10: "Gold | Metals | Materials",
    11: "IPO",
    12: "Legal | Regulation",
    13: "M&A | Investments",
    14: "Macro",
    15: "Markets",
    16: "Politics",
    17: "Personnel Change",
    18: "Stock Commentary",
    19: "Stock Movement",
}

MAX_VOCAB_SIZE = 15000
EPOCHS = 25
BATCH_SIZE = 32


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/validation splits of zeroshot/twitter-financial-news-topic.

    Tries the `datasets` library first (simplest, handles caching); falls
    back to a direct CSV download if `datasets` isn't installed. Either
    way, requires real internet access to huggingface.co.
    """
    try:
        from datasets import load_dataset as hf_load_dataset

        print("Loading via the `datasets` library...")
        ds = hf_load_dataset("zeroshot/twitter-financial-news-topic")
        train_df = ds["train"].to_pandas()
        valid_df = ds["validation"].to_pandas()
    except ImportError:
        print("`datasets` not installed - falling back to direct CSV download.")
        base = (
            "https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic"
            "/resolve/main"
        )
        train_df = pd.read_csv(f"{base}/topic_train.csv")
        valid_df = pd.read_csv(f"{base}/topic_valid.csv")

    print(f"Loaded {len(train_df)} training rows, {len(valid_df)} validation rows.")
    return train_df, valid_df


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def main() -> None:
    train_df, valid_df = load_dataset()

    # Map numeric label -> topic name string BEFORE fitting the label
    # encoder, so label_encoder.pkl's inverse_transform() returns real
    # topic names (e.g. "Earnings") - exactly what nlp_router.py and
    # Phase 2's metadata filter expect, not raw integers.
    train_df["topic"] = train_df["label"].map(ID2LABEL)
    valid_df["topic"] = valid_df["label"].map(ID2LABEL)

    unmapped = train_df["topic"].isna().sum() + valid_df["topic"].isna().sum()
    if unmapped:
        raise ValueError(
            f"{unmapped} rows had a label id outside ID2LABEL's 0-19 range - "
            "the upstream dataset's label scheme may have changed; check "
            "huggingface.co/datasets/zeroshot/twitter-financial-news-topic "
            "before proceeding."
        )

    print("\nApplying nlp_router.py's real preprocessing pipeline to all text...")
    train_df["processed"] = train_df["text"].astype(str).map(preprocess)
    valid_df["processed"] = valid_df["text"].astype(str).map(preprocess)

    # Preprocessing can (rarely) reduce a very short/symbol-only tweet to an
    # empty string - drop those rather than feeding the vectorizer blanks.
    before = len(train_df)
    train_df = train_df[train_df["processed"].str.len() > 0]
    if len(train_df) < before:
        print(f"Dropped {before - len(train_df)} training rows that became empty after preprocessing.")

    print("\nFitting CountVectorizer (Bag-of-Words) on training text...")
    vectorizer = CountVectorizer(max_features=MAX_VOCAB_SIZE)
    X_train = vectorizer.fit_transform(train_df["processed"]).toarray().astype("float32")
    X_valid = vectorizer.transform(valid_df["processed"]).toarray().astype("float32")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("\nFitting LabelEncoder on topic names...")
    label_encoder = LabelEncoder()
    label_encoder.fit(list(ID2LABEL.values()))
    y_train = label_encoder.transform(train_df["topic"])
    y_valid = label_encoder.transform(valid_df["topic"])
    num_classes = len(label_encoder.classes_)
    print(f"Classes ({num_classes}): {list(label_encoder.classes_)}")

    # Class weights: this dataset (like most real financial-news corpora)
    # is imbalanced - some topics (e.g. "Stock Movement") have far more
    # examples than others (e.g. "IPO"). Without weighting, the model can
    # get a deceptively high accuracy just by favoring common classes.
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    print("\nBuilding Keras model...")
    import tensorflow as tf

    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\nTraining...")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=2,
    )

    # --- Honest evaluation on the held-out validation set --- #
    print("\n" + "=" * 72)
    print("VALIDATION SET EVALUATION (real numbers, not assumed)")
    print("=" * 72)
    y_pred_probs = model.predict(X_valid, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    weighted_f1 = f1_score(y_valid, y_pred, average="weighted")
    print(f"\nWeighted F1: {weighted_f1:.4f}")
    print("\nFull classification report:")
    print(
        classification_report(
            y_valid, y_pred, target_names=label_encoder.classes_, zero_division=0
        )
    )

    # --- Save artifacts --- #
    print(f"Saving model to {MODEL_OUT}...")
    model.save(MODEL_OUT)

    print(f"Saving vectorizer to {VECTORIZER_OUT}...")
    with open(VECTORIZER_OUT, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saving label encoder to {LABEL_ENCODER_OUT}...")
    with open(LABEL_ENCODER_OUT, "wb") as f:
        pickle.dump(label_encoder, f)

    print("\nDone. Restart the API (or rebuild the Docker image) to exit Mock Mode.")


if __name__ == "__main__":
    main()
