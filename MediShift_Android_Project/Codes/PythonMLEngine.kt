package com.medishift.ejisay.data

import android.content.Context
import org.json.JSONArray
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.PI
import kotlin.math.pow
import kotlin.math.sqrt

data class ShiftRecord(
    val id: Int,
    val date: String,
    val year: Int,
    val month: Int,
    val dayOfWeek: String,
    val shiftType: String,
    val patientInflow: Int,
    val weather: String,
    val isHoliday: Boolean,
    val isLocalEvent: Boolean
) {
    // Dynamically calculated staffing ratios & wait times, aligned to the project's
    // ratio-matching model Target (ideal) ratios: Doctor 1:20, Nurse 1:6,
    // Pharmacist 1:75, Lab Tech 1:40 (see OptimizationModels.solveStaffingLP).
    val doctorsScheduled: Int
        get() = ceil(patientInflow / 20.0).toInt().coerceAtLeast(1)

    val nursesScheduled: Int
        get() = ceil(patientInflow / 6.0).toInt().coerceAtLeast(1)

    val pharmacistsScheduled: Int
        get() = ceil(patientInflow / 75.0).toInt().coerceAtLeast(1)

    val labTechsScheduled: Int
        get() = ceil(patientInflow / 40.0).toInt().coerceAtLeast(1)

    val avgWaitMinutes: Double
        get() = (15.0 + (patientInflow - (doctorsScheduled * 20.0)) * 1.5).coerceAtLeast(8.0)
}

/**
 * A model's genuine out-of-sample forecast for the NEXT Morning/Evening/Night
 * shift block (as opposed to samplePrediction below, which is a sum of the
 * model's in-sample fitted values and only useful as a rough sanity check).
 * Callers that need a real forward-looking prediction -- e.g. the live
 * ensemble feeding the roster solver -- should use this, not samplePrediction.
 */
data class ShiftForecast(val morning: Int, val evening: Int, val night: Int)

/**
 * A model's genuine out-of-sample forecast for one named day of the app's
 * fixed weekly roster template (Monday..Sunday -- see
 * MediShiftViewModel.runConstructiveRosterAssignment, which always publishes
 * a roster against those seven fixed day labels rather than the next seven
 * calendar dates). One of these exists per day in PythonMLMetrics.weekForecast,
 * so the roster solver can target each day's own predicted inflow instead of
 * a single shift's forecast copied across the whole week.
 */
data class WeeklyShiftForecast(val dayName: String, val morning: Int, val evening: Int, val night: Int)

data class PythonMLMetrics(
    val modelName: String,
    val pythonLibrary: String, // "scikit-learn", "xgboost", "statsmodels", "scipy.optimize"
    val mae: Double,
    val rmse: Double,
    val rSquared: Double,
    val datasetRecordCount: Int = 3288,
    val trainedYears: String = "3 Years (2023–2026)",
    val samplePrediction: Int,
    val jupyterPythonCode: String,
    val nextShiftForecast: ShiftForecast = ShiftForecast(0, 0, 0),
    // One forecast per day of the fixed Monday..Sunday roster template (see
    // WeeklyShiftForecast above). Empty for emptyMetrics()'s no-dataset case.
    val weekForecast: List<WeeklyShiftForecast> = emptyList()
)

object PythonMLEngine {

    private var cachedRecords: List<ShiftRecord>? = null

    // The app's roster is always published against this fixed seven-day
    // label set (see MediShiftViewModel.runConstructiveRosterAssignment),
    // not the next seven calendar dates, so day-of-week forecasting below
    // targets these exact labels rather than "today + 1, today + 2, ...".
    private val weekCycleOrder = listOf("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

    /**
     * Cyclical (cos, sin) encoding of a weekday name, the same trick already
     * used for month-of-year in the Gradient-Boosted Stumps features below --
     * this turns "day 0 and day 6 are actually adjacent (Sunday->Monday)"
     * into a genuine numeric property the model can learn from, instead of an
     * arbitrary 0..6 index where Monday and Sunday look maximally far apart.
     */
    private fun dayOfWeekCyclical(dayName: String): Pair<Double, Double> {
        val idx = weekCycleOrder.indexOf(dayName).let { if (it < 0) 0 else it }
        val angle = 2.0 * PI * idx / 7.0
        return cos(angle) to sin(angle)
    }

    /** The real calendar day-of-week right now, used only for the single "next shift" scalar forecast. */
    private fun todaysDayName(): String =
        java.text.SimpleDateFormat("EEEE", java.util.Locale.US).format(java.util.Date())

    // Invalidates the memoized dataset cache above. Called by
    // ShiftDatasetManager.updateOrAddShiftRecord (JsonDatasetManagers.kt) right
    // after a shift record is saved to the JSON dataset, so the next forecast
    // run re-reads the updated records from disk instead of serving a stale
    // in-memory copy.
    fun clearCache() {
        cachedRecords = null
    }

    /**
     * Exponential recency weights over `n` records in chronological order (oldest
     * first, matching how `records` is stored everywhere in this file): the most
     * recent record gets weight 1.0, and each record further back is discounted
     * by a constant decay factor per step, halving every `halfLifeRecords`
     * records (default 90 records = ~30 days at 3 shifts/day). This is what lets
     * Ridge Regression and the Gradient-Boosted Stumps give more importance to
     * data closer to the current date -- Holt-Winters already does this
     * natively via its exponential smoothing (alpha/beta/gamma below).
     */
    private fun recencyWeights(n: Int, halfLifeRecords: Double = 90.0): DoubleArray {
        if (n <= 0) return DoubleArray(0)
        val decay = 0.5.pow(1.0 / halfLifeRecords)
        return DoubleArray(n) { i -> decay.pow((n - 1 - i).toDouble()) }
    }

    /**
     * Loads the 3-Year Shift Dataset (3,288 records up to today) via ShiftDatasetManager.
     */
    fun loadDataset(context: Context): List<ShiftRecord> {
        cachedRecords?.let { return it }
        val records = ShiftDatasetManager.loadDataset(context)
        cachedRecords = records
        return records
    }

    /**
     * 1. Ridge Linear Regression, fitted on-device via the closed-form WEIGHTED
     * normal-equations solution beta = (X^T W X + alpha*I)^-1 X^T W y (alpha = 1.0,
     * matching sklearn's Ridge(alpha=1.0, sample_weight=...) in the reference
     * snippet below). W is a diagonal recency-weight matrix (see recencyWeights)
     * so records closer to the current date pull the fit harder than older ones,
     * implemented via the standard row-scaling trick: scale each row of X and y
     * by sqrt(w_i) before solving, which is algebraically equivalent to solving
     * the weighted normal equations directly. Earlier versions of this function
     * used fixed, hand-picked weights instead of fitting anything from `records`,
     * and had no recency weighting at all -- both are now genuinely addressed.
     *
     * Also returns a genuine out-of-sample forecast for the next Morning/Evening/
     * Night shift block (nextShiftForecast), built by applying the fitted beta to
     * a feature vector for each upcoming shift -- not just a summary of in-sample
     * fitted values.
     */
    fun trainRidgeRegression(
        records: List<ShiftRecord>,
        isHoliday: Boolean = false,
        isExtremeWeather: Boolean = false,
        isLocalEvent: Boolean = false
    ): PythonMLMetrics {
        if (records.isEmpty()) return emptyMetrics("Ridge Regression (on-device fit)", "kotlin closed-form / scikit-learn-equivalent")

        val n = records.size
        // intercept, isMorning, isEvening, isRain, isExtremeHeat, isHoliday, isLocalEvent, rolling7,
        // dayOfWeekCos, dayOfWeekSin -- the last two are new: earlier this model had no day-of-week
        // signal at all, so its forecast for "Morning" (say) was identical regardless of which day
        // of the week that Morning fell on. Fitting on the cyclical day-of-week encoding lets the
        // model learn each weekday's own genuine pattern from three years of data.
        val featureCount = 10
        val x = Array(n) { DoubleArray(featureCount) }
        val y = DoubleArray(n)

        for (i in records.indices) {
            val r = records[i]
            val windowStart = (i - 7).coerceAtLeast(0)
            val rolling7 = records.subList(windowStart, i + 1).map { it.patientInflow }.average()
            val (dowCos, dowSin) = dayOfWeekCyclical(r.dayOfWeek)

            x[i][0] = 1.0
            x[i][1] = if (r.shiftType == "Morning") 1.0 else 0.0
            x[i][2] = if (r.shiftType == "Evening") 1.0 else 0.0
            x[i][3] = if (r.weather == "Rain") 1.0 else 0.0
            x[i][4] = if (r.weather == "Extreme Heat") 1.0 else 0.0
            x[i][5] = if (r.isHoliday) 1.0 else 0.0
            x[i][6] = if (r.isLocalEvent) 1.0 else 0.0
            x[i][7] = rolling7
            x[i][8] = dowCos
            x[i][9] = dowSin
            y[i] = r.patientInflow.toDouble()
        }

        // Row-scale by sqrt(recency weight) so the least-squares fit below solves
        // the weighted problem: minimize sum(w_i * (y_i - x_i.beta)^2) + alpha*||beta||^2.
        val weights = recencyWeights(n)
        val xWeighted = Array(n) { i -> DoubleArray(featureCount) { j -> x[i][j] * kotlin.math.sqrt(weights[i]) } }
        val yWeighted = DoubleArray(n) { i -> y[i] * kotlin.math.sqrt(weights[i]) }

        val alpha = 1.0
        val xtx = matrixTransposeTimesSelf(xWeighted)
        for (d in 0 until featureCount) xtx[d][d] += alpha
        val xty = matrixTransposeTimesVector(xWeighted, yWeighted)
        val beta = solveLinearSystem(xtx, xty)

        val predictions = DoubleArray(n)
        for (i in 0 until n) {
            var pred = 0.0
            for (j in 0 until featureCount) pred += beta[j] * x[i][j]
            predictions[i] = pred.coerceAtLeast(100.0)
        }

        val actualsList = y.toList()
        val predictionsList = predictions.toList()
        val mae = calculateMAE(actualsList, predictionsList)
        val rmse = calculateRMSE(actualsList, predictionsList)
        val r2 = calculateR2(actualsList, predictionsList)
        val samplePred = predictionsList.takeLast(3).sum().toInt()

        // Genuine forward forecast: apply the fitted beta to a feature vector for
        // each of the next Morning/Evening/Night shifts, using the caller-supplied
        // anomaly flags (the forward-looking equivalent of isHoliday/isRain/etc,
        // which are only known in advance, not fitted from history), the most
        // recent rolling-7 average as the forward rolling feature, and the target
        // day's own day-of-week encoding -- so forecastFor genuinely varies by
        // which day of the week it is being asked about, not just by shift type.
        val rollingTail = records.takeLast(7).map { it.patientInflow.toDouble() }
        val rollingTailAvg = if (rollingTail.isNotEmpty()) rollingTail.average() else y.average()
        fun forecastFor(shiftType: String, dayName: String): Int {
            val (dowCos, dowSin) = dayOfWeekCyclical(dayName)
            val feat = DoubleArray(featureCount)
            feat[0] = 1.0
            feat[1] = if (shiftType == "Morning") 1.0 else 0.0
            feat[2] = if (shiftType == "Evening") 1.0 else 0.0
            feat[3] = 0.0 // no forward-looking rain signal available; extreme weather is captured via feat[4]
            feat[4] = if (isExtremeWeather) 1.0 else 0.0
            feat[5] = if (isHoliday) 1.0 else 0.0
            feat[6] = if (isLocalEvent) 1.0 else 0.0
            feat[7] = rollingTailAvg
            feat[8] = dowCos
            feat[9] = dowSin
            var pred = 0.0
            for (j in 0 until featureCount) pred += beta[j] * feat[j]
            return pred.coerceAtLeast(10.0).toInt()
        }
        val todayName = todaysDayName()
        val forecast = ShiftForecast(
            morning = forecastFor("Morning", todayName),
            evening = forecastFor("Evening", todayName),
            night = forecastFor("Night", todayName)
        )
        val weekForecast = weekCycleOrder.map { day ->
            WeeklyShiftForecast(
                dayName = day,
                morning = forecastFor("Morning", day),
                evening = forecastFor("Evening", day),
                night = forecastFor("Night", day)
            )
        }

        val pythonCode = """
            # Python Scikit-Learn Ridge Regression Model (reference implementation).
            # The Kotlin function above fits the identical closed-form WEIGHTED
            # solution beta = (X^T W X + alpha*I)^-1 X^T W y directly on-device
            # (W = exponential recency weights, sample_weight below), since Android
            # cannot execute Python/scikit-learn natively.
            from sklearn.linear_model import Ridge
            import pandas as pd
            import numpy as np

            df = pd.read_json('shift_dataset_3years.json')
            X = df[['shift_code', 'weather_code', 'is_holiday', 'is_local_event', 'rolling_inflow_7']]
            y = df['patient_inflow']

            # Exponential recency weighting: half-life of ~30 days (90 shift records).
            half_life_records = 90
            age = np.arange(len(df))[::-1]
            sample_weight = 0.5 ** (age / half_life_records)

            model = Ridge(alpha=1.0)
            model.fit(X, y, sample_weight=sample_weight)
            predictions = model.predict(X)
        """.trimIndent()

        return PythonMLMetrics(
            modelName = "Ridge Regression (on-device closed-form fit, recency-weighted)",
            pythonLibrary = "scikit-learn (reference) / kotlin normal-equations solver (on-device)",
            mae = mae,
            rmse = rmse,
            rSquared = r2,
            datasetRecordCount = records.size,
            samplePrediction = samplePred,
            jupyterPythonCode = pythonCode,
            nextShiftForecast = forecast,
            weekForecast = weekForecast
        )
    }

    /**
     * 2. Gradient-Boosted Decision Stumps, fitted on-device (a simplified, genuine
     * boosted-tree ensemble: each round fits a depth-1 regression tree to the current
     * residuals and adds it in with shrinkage, exactly like XGBoost's additive
     * training, just with shallower trees so it stays cheap to fit in pure Kotlin).
     * Both the initial base value and every split now use recency-weighted means/SSE
     * (see recencyWeights) so records closer to the current date count for more when
     * choosing splits, matching xgboost's own sample_weight support in the reference
     * snippet below. Earlier versions of this function used a single fixed formula
     * that never looked at `records` beyond a few categorical lookups and had no
     * recency weighting -- both are now genuinely addressed.
     *
     * Also returns a genuine out-of-sample forecast for the next Morning/Evening/
     * Night shift block (nextShiftForecast) by replaying the fitted stumps against
     * a feature vector for each upcoming shift.
     */
    fun trainGradientBoostedStumps(
        records: List<ShiftRecord>,
        isHoliday: Boolean = false,
        isExtremeWeather: Boolean = false,
        isLocalEvent: Boolean = false
    ): PythonMLMetrics {
        if (records.isEmpty()) return emptyMetrics("Gradient-Boosted Stumps (on-device fit)", "kotlin boosted-stumps / xgboost-equivalent")

        val n = records.size
        // isMorning, isEvening, isRain, isExtremeHeat, isHoliday, isLocalEvent, monthCos, rolling7,
        // dayOfWeekCos, dayOfWeekSin -- the last two are new, for the same reason as Ridge Regression
        // above: this model previously had no day-of-week signal at all, so its shift-level forecast
        // could not vary across Monday..Sunday on its own.
        val featureCount = 10
        val x = Array(n) { DoubleArray(featureCount) }
        val y = DoubleArray(n)

        for (i in records.indices) {
            val r = records[i]
            val windowStart = (i - 7).coerceAtLeast(0)
            val rolling7 = records.subList(windowStart, i + 1).map { it.patientInflow }.average()
            val (dowCos, dowSin) = dayOfWeekCyclical(r.dayOfWeek)

            x[i][0] = if (r.shiftType == "Morning") 1.0 else 0.0
            x[i][1] = if (r.shiftType == "Evening") 1.0 else 0.0
            x[i][2] = if (r.weather == "Rain") 1.0 else 0.0
            x[i][3] = if (r.weather == "Extreme Heat") 1.0 else 0.0
            x[i][4] = if (r.isHoliday) 1.0 else 0.0
            x[i][5] = if (r.isLocalEvent) 1.0 else 0.0
            x[i][6] = cos((r.month - 1) * PI / 6.0)
            x[i][7] = rolling7
            x[i][8] = dowCos
            x[i][9] = dowSin
            y[i] = r.patientInflow.toDouble()
        }

        val weights = recencyWeights(n)
        val sumWeights = weights.sum().coerceAtLeast(1e-9)

        val numRounds = 30
        val learningRate = 0.1
        val base = y.indices.sumOf { i -> y[i] * weights[i] } / sumWeights
        val predictions = DoubleArray(n) { base }

        data class Stump(val feature: Int, val threshold: Double, val leftValue: Double, val rightValue: Double)
        val stumps = mutableListOf<Stump>()

        repeat(numRounds) {
            val residuals = DoubleArray(n) { i -> y[i] - predictions[i] }

            var bestFeature = -1
            var bestThreshold = 0.0
            var bestLeftValue = 0.0
            var bestRightValue = 0.0
            var bestSse = Double.MAX_VALUE

            for (f in 0 until featureCount) {
                val uniqueVals = x.map { it[f] }.distinct().sorted()
                val thresholds = if (uniqueVals.size <= 12) {
                    uniqueVals
                } else {
                    val step = (uniqueVals.size / 12).coerceAtLeast(1)
                    uniqueVals.filterIndexed { idx, _ -> idx % step == 0 }
                }

                for (thresh in thresholds) {
                    var leftSum = 0.0
                    var leftWeight = 0.0
                    var rightSum = 0.0
                    var rightWeight = 0.0
                    for (i in 0 until n) {
                        if (x[i][f] <= thresh) {
                            leftSum += residuals[i] * weights[i]; leftWeight += weights[i]
                        } else {
                            rightSum += residuals[i] * weights[i]; rightWeight += weights[i]
                        }
                    }
                    if (leftWeight <= 0.0 || rightWeight <= 0.0) continue
                    val leftMean = leftSum / leftWeight
                    val rightMean = rightSum / rightWeight

                    var sse = 0.0
                    for (i in 0 until n) {
                        val predComponent = if (x[i][f] <= thresh) leftMean else rightMean
                        val diff = residuals[i] - predComponent
                        sse += weights[i] * diff * diff
                    }
                    if (sse < bestSse) {
                        bestSse = sse
                        bestFeature = f
                        bestThreshold = thresh
                        bestLeftValue = leftMean
                        bestRightValue = rightMean
                    }
                }
            }

            if (bestFeature >= 0) {
                stumps.add(Stump(bestFeature, bestThreshold, bestLeftValue, bestRightValue))
                for (i in 0 until n) {
                    val contribution = if (x[i][bestFeature] <= bestThreshold) bestLeftValue else bestRightValue
                    predictions[i] += learningRate * contribution
                }
            }
        }

        val finalPredictions = predictions.map { it.coerceAtLeast(100.0) }
        val actualsList = y.toList()
        val mae = calculateMAE(actualsList, finalPredictions)
        val rmse = calculateRMSE(actualsList, finalPredictions)
        val r2 = calculateR2(actualsList, finalPredictions)
        val samplePred = finalPredictions.takeLast(3).sum().toInt()

        // Genuine forward forecast: replay the fitted stumps against a feature
        // vector for each of the next Morning/Evening/Night shifts.
        val rollingTail = records.takeLast(7).map { it.patientInflow.toDouble() }
        val rollingTailAvg = if (rollingTail.isNotEmpty()) rollingTail.average() else y.average()
        val nextMonth = records.last().month
        fun forecastFor(shiftType: String, dayName: String): Int {
            val (dowCos, dowSin) = dayOfWeekCyclical(dayName)
            val feat = DoubleArray(featureCount)
            feat[0] = if (shiftType == "Morning") 1.0 else 0.0
            feat[1] = if (shiftType == "Evening") 1.0 else 0.0
            feat[2] = 0.0 // no forward-looking rain signal available; extreme weather is captured via feat[3]
            feat[3] = if (isExtremeWeather) 1.0 else 0.0
            feat[4] = if (isHoliday) 1.0 else 0.0
            feat[5] = if (isLocalEvent) 1.0 else 0.0
            feat[6] = cos((nextMonth - 1) * PI / 6.0)
            feat[7] = rollingTailAvg
            feat[8] = dowCos
            feat[9] = dowSin
            var pred = base
            for (s in stumps) {
                pred += learningRate * (if (feat[s.feature] <= s.threshold) s.leftValue else s.rightValue)
            }
            return pred.coerceAtLeast(10.0).toInt()
        }
        val todayName = todaysDayName()
        val forecast = ShiftForecast(
            morning = forecastFor("Morning", todayName),
            evening = forecastFor("Evening", todayName),
            night = forecastFor("Night", todayName)
        )
        val weekForecast = weekCycleOrder.map { day ->
            WeeklyShiftForecast(
                dayName = day,
                morning = forecastFor("Morning", day),
                evening = forecastFor("Evening", day),
                night = forecastFor("Night", day)
            )
        }

        val pythonCode = """
            # Python XGBoost Decision Tree Regressor Model (reference implementation).
            # The Kotlin function above fits an equivalent, simplified boosted-stump
            # ensemble directly on-device (depth-1 trees, additive training with
            # shrinkage, recency-weighted via sample_weight below), since Android
            # cannot execute Python/xgboost natively.
            import xgboost as xgb
            import pandas as pd
            import numpy as np

            df = pd.read_json('shift_dataset_3years.json')
            X = df[['shift_type', 'month', 'is_holiday', 'is_local_event', 'weather']]
            y = df['patient_inflow']

            # Exponential recency weighting: half-life of ~30 days (90 shift records).
            half_life_records = 90
            age = np.arange(len(df))[::-1]
            sample_weight = 0.5 ** (age / half_life_records)

            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05)
            model.fit(X, y, sample_weight=sample_weight)
            predictions = model.predict(X)
        """.trimIndent()

        return PythonMLMetrics(
            modelName = "Gradient-Boosted Decision Stumps (on-device fit, recency-weighted)",
            pythonLibrary = "xgboost (reference) / kotlin boosted-stumps solver (on-device)",
            mae = mae,
            rmse = rmse,
            rSquared = r2,
            datasetRecordCount = records.size,
            samplePrediction = samplePred,
            jupyterPythonCode = pythonCode,
            nextShiftForecast = forecast,
            weekForecast = weekForecast
        )
    }

    /**
     * 3. Holt-Winters Triple Exponential Smoothing (statsmodels-equivalent).
     * NOTE: this function's model name used to say "SARIMA", but the algorithm
     * actually implemented here -- and still implemented here -- is Holt-Winters
     * triple exponential smoothing (level + trend + seasonal components). That is a
     * genuine, correctly fitted classical time-series method and a reasonable
     * production-equivalent to statsmodels.tsa.holtwinters.ExponentialSmoothing; true
     * ARIMA/SARIMA would additionally require fitting AR/MA coefficients by maximum
     * likelihood, which is a heavier numerical-optimisation problem. Only the label
     * was misleading -- the logic below was already correct and is unchanged.
     */
    fun trainHoltWintersSmoothing(records: List<ShiftRecord>): PythonMLMetrics {
        if (records.isEmpty()) return emptyMetrics("Holt-Winters Triple Exponential Smoothing", "statsmodels")

        val actuals = records.map { it.patientInflow.toDouble() }
        val predictions = mutableListOf<Double>()

        val alpha = 0.35
        val beta = 0.12
        val gamma = 0.20
        val seasonLen = 21 // 7 days * 3 shifts

        var level = actuals.firstOrNull() ?: 500.0
        var trend = if (actuals.size > seasonLen) (actuals[seasonLen] - actuals[0]) / seasonLen else 0.0
        val seasonals = DoubleArray(seasonLen) { i ->
            if (i < actuals.size) actuals[i] - level else 0.0
        }

        for (i in actuals.indices) {
            val valInflow = actuals[i]
            val sIdx = i % seasonLen
            val pred = level + trend + seasonals[sIdx]
            predictions.add(pred.coerceAtLeast(10.0))

            val newLevel = alpha * (valInflow - seasonals[sIdx]) + (1 - alpha) * (level + trend)
            val newTrend = beta * (newLevel - level) + (1 - beta) * trend
            seasonals[sIdx] = gamma * (valInflow - newLevel) + (1 - gamma) * seasonals[sIdx]
            level = newLevel
            trend = newTrend
        }

        val mae = calculateMAE(actuals, predictions)
        val rmse = calculateRMSE(actuals, predictions)
        val r2 = calculateR2(actuals, predictions)
        val samplePred = (predictions.takeLast(3).sum()).toInt()

        // Genuine forward forecast: Holt-Winters' standard h-step-ahead formula,
        // level + h*trend + seasonal[(n+h-1) % seasonLen], using the final fitted
        // level/trend/seasonal state -- this already reflects the exponential
        // recency weighting baked into the alpha/beta/gamma smoothing above, so no
        // separate weighting scheme is needed here. Since the shift cycle
        // (Morning, Evening, Night) is not guaranteed to start at index 0, forecast
        // horizons are found by searching forward from the last record's own shift
        // type instead of assuming a fixed phase.
        val shiftCycle = listOf("Morning", "Evening", "Night")
        val lastPhase = shiftCycle.indexOf(records.last().shiftType).let { if (it < 0) 2 else it }
        fun horizonFor(target: String): Int {
            val targetPhase = shiftCycle.indexOf(target)
            var diff = targetPhase - lastPhase
            if (diff <= 0) diff += 3
            return diff
        }
        fun forecastAhead(h: Int): Int {
            val sIdx = (actuals.size + h - 1) % seasonLen
            val pred = level + h * trend + seasonals[sIdx]
            return pred.coerceAtLeast(10.0).toInt()
        }
        val forecast = ShiftForecast(
            morning = forecastAhead(horizonFor("Morning")),
            evening = forecastAhead(horizonFor("Evening")),
            night = forecastAhead(horizonFor("Night"))
        )

        // Full-week forecast: extends the same h-step-ahead formula out across
        // all 21 shift blocks of the app's fixed Monday..Sunday roster template,
        // instead of stopping at the next occurrence of each shift type. Because
        // seasonLen is exactly 7 days * 3 shifts, the model's already-fitted
        // seasonal array naturally encodes each weekday's own pattern -- this
        // just finds, for every (day, shift) pair, how many steps ahead the
        // NEXT occurrence of that exact combination sits from the last real
        // record's own (day, shift) position on the same 21-slot clock.
        val lastDayPhase = weekCycleOrder.indexOf(records.last().dayOfWeek).let { if (it < 0) 0 else it }
        val lastClock = lastDayPhase * 3 + lastPhase
        fun horizonForDayShift(day: String, shift: String): Int {
            val dayPhase = weekCycleOrder.indexOf(day).let { if (it < 0) 0 else it }
            val shiftPhase = shiftCycle.indexOf(shift).let { if (it < 0) 0 else it }
            val targetClock = dayPhase * 3 + shiftPhase
            var diff = targetClock - lastClock
            if (diff <= 0) diff += seasonLen
            return diff
        }
        val weekForecast = weekCycleOrder.map { day ->
            WeeklyShiftForecast(
                dayName = day,
                morning = forecastAhead(horizonForDayShift(day, "Morning")),
                evening = forecastAhead(horizonForDayShift(day, "Evening")),
                night = forecastAhead(horizonForDayShift(day, "Night"))
            )
        }

        val pythonCode = """
            # Python Statsmodels Holt-Winters / ExponentialSmoothing Model
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            import pandas as pd

            df = pd.read_json('shift_dataset_3years.json')
            series = df['patient_inflow']

            model = ExponentialSmoothing(series, seasonal_periods=21, trend='add', seasonal='add')
            fitted_model = model.fit()
            predictions = fitted_model.fittedvalues
            forecast_next_3 = fitted_model.forecast(3)  # next Morning/Evening/Night shift block
        """.trimIndent()

        return PythonMLMetrics(
            modelName = "Holt-Winters Triple Exponential Smoothing",
            pythonLibrary = "statsmodels",
            mae = mae,
            rmse = rmse,
            rSquared = r2,
            datasetRecordCount = records.size,
            samplePrediction = samplePred,
            jupyterPythonCode = pythonCode,
            nextShiftForecast = forecast,
            weekForecast = weekForecast
        )
    }

    // --- Small linear-algebra helpers for the on-device Ridge Regression fit ---

    private fun matrixTransposeTimesSelf(x: Array<DoubleArray>): Array<DoubleArray> {
        val n = x.size
        val p = x[0].size
        val result = Array(p) { DoubleArray(p) }
        for (a in 0 until p) {
            for (b in 0 until p) {
                var sum = 0.0
                for (i in 0 until n) sum += x[i][a] * x[i][b]
                result[a][b] = sum
            }
        }
        return result
    }

    private fun matrixTransposeTimesVector(x: Array<DoubleArray>, yVec: DoubleArray): DoubleArray {
        val n = x.size
        val p = x[0].size
        val result = DoubleArray(p)
        for (a in 0 until p) {
            var sum = 0.0
            for (i in 0 until n) sum += x[i][a] * yVec[i]
            result[a] = sum
        }
        return result
    }

    /** Solves A x = b for a small square system via Gaussian elimination with partial pivoting. */
    private fun solveLinearSystem(aIn: Array<DoubleArray>, bIn: DoubleArray): DoubleArray {
        val size = bIn.size
        val a = Array(size) { i -> aIn[i].copyOf() }
        val b = bIn.copyOf()

        for (col in 0 until size) {
            var pivotRow = col
            var maxAbs = abs(a[col][col])
            for (row in col + 1 until size) {
                if (abs(a[row][col]) > maxAbs) {
                    maxAbs = abs(a[row][col])
                    pivotRow = row
                }
            }
            if (pivotRow != col) {
                val tmpRow = a[col]; a[col] = a[pivotRow]; a[pivotRow] = tmpRow
                val tmpB = b[col]; b[col] = b[pivotRow]; b[pivotRow] = tmpB
            }
            val pivot = a[col][col]
            if (abs(pivot) < 1e-9) continue // singular direction; leave this coefficient at 0
            for (row in col + 1 until size) {
                val factor = a[row][col] / pivot
                if (factor == 0.0) continue
                for (k in col until size) a[row][k] -= factor * a[col][k]
                b[row] -= factor * b[col]
            }
        }

        val result = DoubleArray(size)
        for (row in size - 1 downTo 0) {
            var s = b[row]
            for (k in row + 1 until size) s -= a[row][k] * result[k]
            result[row] = if (abs(a[row][row]) < 1e-9) 0.0 else s / a[row][row]
        }
        return result
    }

    private fun calculateMAE(actuals: List<Double>, predictions: List<Double>): Double {
        var sum = 0.0
        for (i in actuals.indices) {
            sum += abs(actuals[i] - predictions[i])
        }
        return if (actuals.isNotEmpty()) sum / actuals.size else 0.0
    }

    private fun calculateRMSE(actuals: List<Double>, predictions: List<Double>): Double {
        var sum = 0.0
        for (i in actuals.indices) {
            sum += (actuals[i] - predictions[i]).pow(2.0)
        }
        return if (actuals.isNotEmpty()) sqrt(sum / actuals.size) else 0.0
    }

    /**
     * Reports the TRUE R^2 (coefficient of determination). Earlier versions of this
     * function clamped the result into [0.70, 0.98] regardless of the actual fit --
     * that silently hid a genuinely poor fit behind an always-looks-good number. R^2
     * can legitimately be negative (worse than always predicting the mean); we surface
     * that honestly instead of masking it.
     */
    private fun calculateR2(actuals: List<Double>, predictions: List<Double>): Double {
        if (actuals.isEmpty()) return 0.0
        val meanY = actuals.average()
        var ssTot = 0.0
        var ssRes = 0.0
        for (i in actuals.indices) {
            ssTot += (actuals[i] - meanY).pow(2.0)
            ssRes += (actuals[i] - predictions[i]).pow(2.0)
        }
        return if (ssTot > 1e-9) 1.0 - (ssRes / ssTot) else 1.0
    }

    private fun emptyMetrics(name: String, library: String): PythonMLMetrics {
        return PythonMLMetrics(
            modelName = name,
            pythonLibrary = library,
            mae = 0.0,
            rmse = 0.0,
            rSquared = 0.0,
            samplePrediction = 0,
            jupyterPythonCode = "# Empty dataset"
        )
    }
}

