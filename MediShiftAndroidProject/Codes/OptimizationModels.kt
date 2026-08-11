package com.example.data

import kotlin.math.sin
import kotlin.math.cos
import kotlin.math.PI

/**
 * Dedicated Math Engine for MediShift.
 * Implements:
 * 1. A closed-form, Integer-Programming-equivalent staffing calculator
 *    (solveStaffingLP) that chooses, per category, how many staff to roster so
 *    the resulting patient-to-staff ratio lands as close as possible to an
 *    IDEAL "Target" ratio, never dropping below a hard-floor "Good" ratio.
 *    Revision note: this REPLACES the project's earlier objective of
 *    minimizing individual working hours. That old objective only ever pushed
 *    staffing down toward the bare-minimum safety floor, since every extra
 *    staff member cost the objective something. The new objective is
 *    symmetric -- overstaffing past the ideal costs just as much as
 *    understaffing below it -- so the model is no longer trying to be cheap,
 *    it is trying to be ACCURATE to the ideal ratio the hospital actually
 *    wants. The formula n*_c = min(ideal_c, pool(c)) used below IS the true
 *    optimum of that objective -- no iterative solver is needed because each
 *    category is independent and ideal_c >= minSafe_c always holds (every
 *    Target ratio in this project is <= its Good ratio).
 * 2. A lightweight, single-formula heuristic forecast (trainAndPredictEnsemble)
 *    for next-week patient inflow. NOTE: despite the field names below
 *    (ridgeRegressionPred / gradientBoostedPred / holtWintersPred, kept for UI/API
 *    compatibility), this is ONE blended recency + momentum + seasonality +
 *    anomaly formula, not three independently trained models. For genuinely
 *    separate, data-fitted Linear Regression / Gradient-Boosted-Tree /
 *    Holt-Winters models, see PythonMLEngine.kt.
 */
object OptimizationModels {

    // --- STAFFING RATIO-MATCHING OPTIMIZATION ENGINE ---
    data class StaffingLPInput(
        val predictedPatients: Int,
        val availableDoctors: Int, // Limited doctors available
        val availableNurses: Int,  // Limited nurses available
        val availablePharmacists: Int,
        val availableLabTechs: Int,
        // "Good" ratio = loosest patient-to-staff ratio still clinically acceptable.
        // This is a hard floor -- staffing must never fall below it.
        val doctorGoodRatio: Double = 50.0,
        val nurseGoodRatio: Double = 20.0,
        val pharmacistGoodRatio: Double = 100.0,
        val labTechGoodRatio: Double = 100.0,
        // "Target" ratio = the ideal patient-to-staff ratio the hospital actually
        // wants to achieve. This is what the model actively aims for. Always
        // <= its corresponding Good ratio in this project.
        val doctorTargetRatio: Double = 20.0,
        val nurseTargetRatio: Double = 6.0,
        val pharmacistTargetRatio: Double = 75.0,
        val labTechTargetRatio: Double = 40.0,
        val doctorHours: Double = 40.0,
        val nurseHours: Double = 36.0,
        val pharmacistHours: Double = 40.0,
        val labTechHours: Double = 38.0,
        val maxOperatingBudget: Double = 150000.0 // Operating budget constraint
    )

    data class StaffingLPResult(
        val doctors: Int,
        val nurses: Int,
        val pharmacists: Int,
        val labTechs: Int,
        val idealDoctors: Int,      // ceil(patients / targetRatio) -- what the model is aiming for
        val idealNurses: Int,
        val idealPharmacists: Int,
        val idealLabTechs: Int,
        val minSafeDoctors: Int,    // ceil(patients / goodRatio) -- the hard safety floor
        val minSafeNurses: Int,
        val minSafePharmacists: Int,
        val minSafeLabTechs: Int,
        val deviationDoctors: Int,  // |chosen - ideal| for this category
        val deviationNurses: Int,
        val deviationPharmacists: Int,
        val deviationLabTechs: Int,
        val totalDeviation: Int,    // objective value: sum of all four deviations
        val totalCost: Double, // total hours scheduled (kept for backward-compatible callers)
        val totalHours: Double, // total hours
        val totalLaborCost: Double, // Estimated labor budget cost
        val isQualityCompromised: Boolean, // true if any category falls below its safety floor
        val isWithinBudget: Boolean,
        val doctorRatioText: String,
        val nurseRatioText: String,
        val pharmacistRatioText: String,
        val labTechRatioText: String,
        val statusMessage: String
    )

    /**
     * Solves the category-wise ratio-matching Integer Program: for each staff
     * category, choose a headcount that lands as close as possible to the ideal
     * "Target" ratio staffing level, without ever dropping below the "Good" ratio
     * safety floor (subject to the available pool). Objective: minimize the total
     * absolute deviation from the ideal, summed across all four categories --
     *   minimize  sum_c |n_c - ideal_c|
     *   subject to  minSafe_c <= n_c <= pool(c)
     * This REPLACES the project's earlier objective (minimize total working
     * hours), which only ever pushed staffing toward the bare minimum.
     *
     * Closed-form solution -- no iterative solver needed: because each category's
     * constraints and objective term only involve that category's own variable,
     * the four categories are fully independent, and because
     * ideal_c >= minSafe_c always holds here (every Target ratio is <= its Good
     * ratio), the true optimum is simply:
     *   n*_c = min(ideal_c, pool(c))
     * i.e. staff as close to the ideal as the available pool allows. If the pool
     * can't even reach minSafe_c, that's a genuine staffing deficit, flagged via
     * isQualityCompromised rather than silently reported as just "deviation."
     */
    fun solveStaffingLP(input: StaffingLPInput): StaffingLPResult {
        val p = input.predictedPatients

        // Hard safety floor per category, from the Good ratio.
        val minSafeDocs = kotlin.math.ceil(p / input.doctorGoodRatio).toInt().coerceAtLeast(1)
        val minSafeNurses = kotlin.math.ceil(p / input.nurseGoodRatio).toInt().coerceAtLeast(1)
        val minSafePharmacists = kotlin.math.ceil(p / input.pharmacistGoodRatio).toInt().coerceAtLeast(1)
        val minSafeLabTechs = kotlin.math.ceil(p / input.labTechGoodRatio).toInt().coerceAtLeast(1)

        // Aspirational headcount per category, from the Target ratio. Always
        // coerced at/above minSafe so ideal_c >= minSafe_c is guaranteed even if a
        // future ratio edit accidentally violates Target <= Good for one category.
        val idealDocs = kotlin.math.ceil(p / input.doctorTargetRatio).toInt().coerceAtLeast(minSafeDocs)
        val idealNurses = kotlin.math.ceil(p / input.nurseTargetRatio).toInt().coerceAtLeast(minSafeNurses)
        val idealPharmacists = kotlin.math.ceil(p / input.pharmacistTargetRatio).toInt().coerceAtLeast(minSafePharmacists)
        val idealLabTechs = kotlin.math.ceil(p / input.labTechTargetRatio).toInt().coerceAtLeast(minSafeLabTechs)

        // Closed-form optimum: staff as close to ideal as the pool allows.
        val finalDocs = idealDocs.coerceAtMost(input.availableDoctors)
        val finalNurses = idealNurses.coerceAtMost(input.availableNurses)
        val finalPharmacists = idealPharmacists.coerceAtMost(input.availablePharmacists)
        val finalLabTechs = idealLabTechs.coerceAtMost(input.availableLabTechs)

        val deviationDocs = kotlin.math.abs(finalDocs - idealDocs)
        val deviationNurses = kotlin.math.abs(finalNurses - idealNurses)
        val deviationPharmacists = kotlin.math.abs(finalPharmacists - idealPharmacists)
        val deviationLabTechs = kotlin.math.abs(finalLabTechs - idealLabTechs)
        val totalDeviation = deviationDocs + deviationNurses + deviationPharmacists + deviationLabTechs

        // Genuine deficit only when even the safety floor can't be met -- distinct
        // from an unavoidable-but-safe gap from the ideal.
        val isQualityCompromised = finalDocs < minSafeDocs || finalNurses < minSafeNurses ||
            finalPharmacists < minSafePharmacists || finalLabTechs < minSafeLabTechs

        val statusMessage = if (!isQualityCompromised) {
            "Staffing matched as closely as possible to the ideal Target ratios. Total deviation from ideal across all four categories: $totalDeviation staff. Every category is at or above its Good-ratio safety floor."
        } else {
            val shortages = mutableListOf<String>()
            if (finalDocs < minSafeDocs) shortages.add("${minSafeDocs - finalDocs} Doctors")
            if (finalNurses < minSafeNurses) shortages.add("${minSafeNurses - finalNurses} Nurses")
            if (finalPharmacists < minSafePharmacists) shortages.add("${minSafePharmacists - finalPharmacists} Pharmacists")
            if (finalLabTechs < minSafeLabTechs) shortages.add("${minSafeLabTechs - finalLabTechs} Lab Techs")
            "STAFF DEFICIT DETECTED: available pool cannot even reach the Good-ratio safety floor for $p patients. Shortage: ${shortages.joinToString(", ")}. Total deviation from ideal: $totalDeviation staff."
        }

        val totalHours = (finalDocs * input.doctorHours) +
                         (finalNurses * input.nurseHours) +
                         (finalPharmacists * input.pharmacistHours) +
                         (finalLabTechs * input.labTechHours)

        // Estimated labor cost (Doctor: ₹1200/h, Nurse: ₹500/h, Phar: ₹600/h, Lab: ₹450/h)
        val laborCost = (finalDocs * input.doctorHours * 1200.0) +
                        (finalNurses * input.nurseHours * 500.0) +
                        (finalPharmacists * input.pharmacistHours * 600.0) +
                        (finalLabTechs * input.labTechHours * 450.0)

        val isWithinBudget = laborCost <= input.maxOperatingBudget || input.maxOperatingBudget <= 0

        val docRatioStr = "1:${String.format("%.1f", p.toDouble() / finalDocs.coerceAtLeast(1))} (Target 1:${input.doctorTargetRatio.toInt()}, Good 1:${input.doctorGoodRatio.toInt()})"
        val nurseRatioStr = "1:${String.format("%.1f", p.toDouble() / finalNurses.coerceAtLeast(1))} (Target 1:${input.nurseTargetRatio.toInt()}, Good 1:${input.nurseGoodRatio.toInt()})"
        val pharRatioStr = "1:${String.format("%.1f", p.toDouble() / finalPharmacists.coerceAtLeast(1))} (Target 1:${input.pharmacistTargetRatio.toInt()}, Good 1:${input.pharmacistGoodRatio.toInt()})"
        val labRatioStr = "1:${String.format("%.1f", p.toDouble() / finalLabTechs.coerceAtLeast(1))} (Target 1:${input.labTechTargetRatio.toInt()}, Good 1:${input.labTechGoodRatio.toInt()})"

        return StaffingLPResult(
            doctors = finalDocs,
            nurses = finalNurses,
            pharmacists = finalPharmacists,
            labTechs = finalLabTechs,
            idealDoctors = idealDocs,
            idealNurses = idealNurses,
            idealPharmacists = idealPharmacists,
            idealLabTechs = idealLabTechs,
            minSafeDoctors = minSafeDocs,
            minSafeNurses = minSafeNurses,
            minSafePharmacists = minSafePharmacists,
            minSafeLabTechs = minSafeLabTechs,
            deviationDoctors = deviationDocs,
            deviationNurses = deviationNurses,
            deviationPharmacists = deviationPharmacists,
            deviationLabTechs = deviationLabTechs,
            totalDeviation = totalDeviation,
            totalCost = totalHours,
            totalHours = totalHours,
            totalLaborCost = laborCost,
            isQualityCompromised = isQualityCompromised,
            isWithinBudget = isWithinBudget,
            doctorRatioText = docRatioStr,
            nurseRatioText = nurseRatioStr,
            pharmacistRatioText = pharRatioStr,
            labTechRatioText = labRatioStr,
            statusMessage = statusMessage
        )
    }

    // --- PER-SHIFT AGGREGATE OPTIMALITY (CAPACITY PLANNING ACROSS ALL SHIFT BLOCKS) ---
    data class DailyStaffingPlan(
        val morning: StaffingLPResult,
        val evening: StaffingLPResult,
        val night: StaffingLPResult,
        val totalDeviationAllShifts: Int,     // objective value summed across all 3 shift blocks
        val anyShiftCompromised: Boolean,     // true if ANY shift block falls below its safety floor
        val compromisedShifts: List<String>,  // names of shift blocks that are below their safety floor
        val summary: String
    )

    /**
     * Applies the same category-wise ratio-matching optimum (solveStaffingLP) to
     * EVERY shift block of the day -- Morning, Evening, Night -- so the objective
     * function "minimize sum_c |n_c - ideal_c|" is genuinely satisfied at every
     * shift, not just once for the day as a whole.
     *
     * Each shift block is solved against its OWN predicted patient count (the
     * per-shift split of the day's forecast) but against the SAME available
     * staff pool -- this function answers "how many staff of each category
     * should be on duty during this shift block," which is a capacity-planning
     * question distinct from the person-level roster assignment in
     * runConstructiveRosterAssignment (which partitions a single day's pool of people across
     * shifts so nobody is double-booked). Here, "available" legitimately means
     * the same total headcount pool for each shift, because different people
     * work each shift block.
     */
    fun solveStaffingLPAllShifts(
        morningPatients: Int,
        eveningPatients: Int,
        nightPatients: Int,
        availableDoctors: Int,
        availableNurses: Int,
        availablePharmacists: Int,
        availableLabTechs: Int,
        doctorGoodRatio: Double = 50.0,
        nurseGoodRatio: Double = 20.0,
        pharmacistGoodRatio: Double = 100.0,
        labTechGoodRatio: Double = 100.0,
        doctorTargetRatio: Double = 20.0,
        nurseTargetRatio: Double = 6.0,
        pharmacistTargetRatio: Double = 75.0,
        labTechTargetRatio: Double = 40.0,
        doctorHours: Double = 40.0,
        nurseHours: Double = 36.0,
        pharmacistHours: Double = 40.0,
        labTechHours: Double = 38.0,
        maxOperatingBudget: Double = 150000.0
    ): DailyStaffingPlan {
        fun solveFor(shiftPatients: Int): StaffingLPResult = solveStaffingLP(
            StaffingLPInput(
                predictedPatients = shiftPatients,
                availableDoctors = availableDoctors,
                availableNurses = availableNurses,
                availablePharmacists = availablePharmacists,
                availableLabTechs = availableLabTechs,
                doctorGoodRatio = doctorGoodRatio,
                nurseGoodRatio = nurseGoodRatio,
                pharmacistGoodRatio = pharmacistGoodRatio,
                labTechGoodRatio = labTechGoodRatio,
                doctorTargetRatio = doctorTargetRatio,
                nurseTargetRatio = nurseTargetRatio,
                pharmacistTargetRatio = pharmacistTargetRatio,
                labTechTargetRatio = labTechTargetRatio,
                doctorHours = doctorHours,
                nurseHours = nurseHours,
                pharmacistHours = pharmacistHours,
                labTechHours = labTechHours,
                // Per-shift budget share -- one third of the daily operating
                // budget is available to each shift block's labor cost check.
                maxOperatingBudget = maxOperatingBudget / 3.0
            )
        )

        val morningResult = solveFor(morningPatients)
        val eveningResult = solveFor(eveningPatients)
        val nightResult = solveFor(nightPatients)

        val totalDeviationAllShifts = morningResult.totalDeviation + eveningResult.totalDeviation + nightResult.totalDeviation

        val compromisedShifts = mutableListOf<String>()
        if (morningResult.isQualityCompromised) compromisedShifts.add("Morning")
        if (eveningResult.isQualityCompromised) compromisedShifts.add("Evening")
        if (nightResult.isQualityCompromised) compromisedShifts.add("Night")
        val anyShiftCompromised = compromisedShifts.isNotEmpty()

        val summary = if (!anyShiftCompromised) {
            "Optimality condition satisfied at every shift block. Combined deviation from ideal across Morning, Evening, and Night: $totalDeviationAllShifts staff."
        } else {
            "Shift-level staffing deficit detected in: ${compromisedShifts.joinToString(", ")}. Combined deviation from ideal across all shift blocks: $totalDeviationAllShifts staff."
        }

        return DailyStaffingPlan(
            morning = morningResult,
            evening = eveningResult,
            night = nightResult,
            totalDeviationAllShifts = totalDeviationAllShifts,
            anyShiftCompromised = anyShiftCompromised,
            compromisedShifts = compromisedShifts,
            summary = summary
        )
    }

    // --- ENSEMBLE FORECASTING ENGINE ---

    /**
     * One day's forecast within the app's fixed Monday..Sunday roster template
     * (see MediShiftViewModel.runConstructiveRosterAssignment). ForecastEnsembleResult.weeklyForecast
     * holds exactly seven of these, one per roster day, so the roster solver can
     * target each day's own predicted inflow instead of reusing a single day's
     * number for the whole week.
     */
    data class DayForecast(
        val day: String,
        val morning: Int,
        val evening: Int,
        val night: Int
    ) {
        val total: Int get() = morning + evening + night
    }

    data class ForecastEnsembleResult(
        val ridgeRegressionPred: Double,
        val gradientBoostedPred: Double,
        val holtWintersPred: Double,
        val ensemblePred: Int,
        val morningPred: Int = 0,
        val eveningPred: Int = 0,
        val nightPred: Int = 0,
        val fitConfidence: String,
        val trendAnalysis: String = "",
        // Genuine blend shares (sum to ~100), so the UI can show what the
        // ensemble actually did instead of a hardcoded, unrelated number.
        // isRealEnsemble = false means no per-shift dataset was available, so
        // the three *Pred values above are heuristic variants, not
        // independently fitted models, and no real weight applies.
        val isRealEnsemble: Boolean = false,
        val ridgeWeightPercent: Int = 0,
        val gradientBoostedWeightPercent: Int = 0,
        val holtWintersWeightPercent: Int = 0,
        val heuristicWeightPercent: Int = 100,
        // One forecast per day of the fixed roster week (Monday..Sunday), used
        // by runConstructiveRosterAssignment to give each day its own staffing
        // target. morningPred/eveningPred/nightPred above remain a single
        // "right now" snapshot for the quick-glance UI cards; this is the
        // per-day breakdown that actually drives the roster.
        val weeklyForecast: List<DayForecast> = emptyList()
    )

    /**
     * Hospital capacity forecasting ensemble for next-shift patient inflow.
     *
     * When `shiftRecords` is supplied (the 3-year per-shift dataset from
     * PythonMLEngine/ShiftDatasetManager), this is a REAL ensemble: it trains the
     * three genuinely data-fitted, recency-weighted models in PythonMLEngine.kt
     * (Ridge Regression, Gradient-Boosted Stumps, Holt-Winters) and blends their
     * actual forward forecasts with the fast heuristic below. Previously this
     * function only ever computed ONE blended recency + momentum + seasonality +
     * anomaly heuristic and mislabeled three differently-weighted views of that
     * SAME number as if they were ridgeRegressionPred / gradientBoostedPred / holtWintersPred
     * -- independently-fitted models existed in PythonMLEngine.kt but were never
     * wired into the live prediction actually driving the roster solver, only
     * into a separate ML Dashboard display screen. That gap is what this
     * function now closes: when shiftRecords is available, those three field
     * names are genuinely true again.
     *
     * When `shiftRecords` is null/empty (e.g. a caller without dataset access),
     * this falls back to the original single-formula heuristic unchanged, so
     * behavior stays backward compatible.
     */
    fun trainAndPredictEnsemble(
        history: List<HistoricalInflow>,
        isHoliday: Boolean = false,
        isExtremeWeather: Boolean = false,
        isLocalEvent: Boolean = false,
        shiftRecords: List<ShiftRecord>? = null
    ): ForecastEnsembleResult {
        val values = history.map { it.patientCount.toDouble() }
        val size = values.size

        // 1. Long-term historical baseline from dynamic dataset
        val baselineAvg = if (size > 0) values.average() else 1550.0

        // 2. Recency Bias: give higher weight to receptionist logged intake & recent data points.
        // Uses a 7-day exponentially decaying window (half-life ~3 days) rather than a flat
        // average of the last 3 raw points -- a flat 3-point average let a single unusual day
        // (a receptionist mis-log, one abnormally busy or quiet day) carry a full third of this
        // term by itself, which then propagated into every downstream prediction. Smoothing over
        // a wider, still recency-biased window keeps today's data influential without letting one
        // data point swing the whole forecast.
        val recentWindow = if (size >= 7) values.takeLast(7) else (if (size > 0) values else listOf(1550.0))
        val windowSize = recentWindow.size
        val recentDecay = Math.pow(0.5, 1.0 / 3.0) // half-life of ~3 days within the window
        var recentWeightedSum = 0.0
        var recentWeightTotal = 0.0
        for (i in recentWindow.indices) {
            val w = Math.pow(recentDecay, (windowSize - 1 - i).toDouble())
            recentWeightedSum += recentWindow[i] * w
            recentWeightTotal += w
        }
        val recentAvg = if (recentWeightTotal > 0) recentWeightedSum / recentWeightTotal else recentWindow.average()

        // Weight: 75% recency (receptionist logged data), 25% long-term dataset baseline
        val recencyBiasedBase = (recentAvg * 0.75) + (baselineAvg * 0.25)

        // 3. Non-linear Trend and Momentum Changes
        val momentumChange = if (size >= 3) {
            val w3 = values[size - 3]
            val w2 = values[size - 2]
            val w1 = values[size - 1]
            val pctChange1 = (w2 - w3) / w3.coerceAtLeast(1.0)
            val pctChange2 = (w1 - w2) / w2.coerceAtLeast(1.0)
            (pctChange1 + pctChange2) / 2.0
        } else 0.0

        val momentumMultiplier = 1.0 + momentumChange.coerceIn(-0.20, 0.20)

        // Cyclical and Seasonal Trends: weekly periodicity sinusoidal adjustments
        val cyclicalFactor = 1.0 + 0.06 * kotlin.math.sin((size.toDouble() * 2.0 * kotlin.math.PI) / 7.0)

        // 4. Anomaly processing (Holidays, Extreme Weather, Local Events)
        var anomalyMultiplier = 1.0
        if (isHoliday) anomalyMultiplier += -0.12
        if (isExtremeWeather) anomalyMultiplier += -0.22
        if (isLocalEvent) anomalyMultiplier += 0.15

        // 4b. Day-of-week seasonality factor for the heuristic component,
        // derived from the day-level historical dataset: how much higher or
        // lower a given weekday's average patient count runs relative to the
        // overall average. This is what lets the heuristic's contribution to
        // the week actually vary by day instead of repeating one flat number
        // seven times -- the only weekly-shaped term above, cyclicalFactor,
        // moves with the sheer COUNT of historical records fed in, not with
        // which real weekday is being forecast, so on its own it could not
        // tell a Monday from a Thursday. A weekday needs at least 3
        // historical samples before its factor is trusted (fewer than that
        // is too easily set by a single outlier day, which is exactly the
        // kind of single-day dominance this fix is meant to reduce), and
        // every factor is clamped to [0.8, 1.25] so no one day's history can
        // swing a shift target by more than about a quarter.
        val weekDayNames = listOf("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        fun dayNameForDateStr(dateStr: String): String? = try {
            val parsed = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).parse(dateStr)
            parsed?.let { java.text.SimpleDateFormat("EEEE", java.util.Locale.US).format(it) }
        } catch (e: Exception) {
            null
        }
        val byWeekday: Map<String, List<Double>> = history.mapNotNull { rec ->
            dayNameForDateStr(rec.date)?.let { dayName -> dayName to rec.patientCount.toDouble() }
        }.groupBy({ it.first }, { it.second })
        val dayOfWeekFactor: Map<String, Double> = weekDayNames.associateWith { day ->
            val samples = byWeekday[day] ?: emptyList()
            if (samples.size < 3 || baselineAvg <= 0.0) {
                1.0
            } else {
                (samples.average() / baselineAvg).coerceIn(0.8, 1.25)
            }
        }

        // Heuristic prediction components (always computed as the fallback/blend
        // partner below -- it reacts instantly to the isHoliday/isExtremeWeather/
        // isLocalEvent toggles the user just flipped, which is valuable even when
        // the real models are available, since those anomaly flags are sparse in
        // the training set and a fresh toggle may not yet be well-represented
        // in the fitted models).
        val ridgeTrendPred = recencyBiasedBase * momentumMultiplier
        val nonLinearBoostedPred = recencyBiasedBase * momentumMultiplier * cyclicalFactor
        val seasonalHoltWintersPred = recencyBiasedBase * cyclicalFactor * anomalyMultiplier

        val heuristicEnsemble = (recencyBiasedBase * momentumMultiplier * cyclicalFactor * anomalyMultiplier).toInt().coerceAtLeast(100)
        val heuristicMorning = (heuristicEnsemble * 0.45).toInt().coerceAtLeast(10)
        val heuristicEvening = (heuristicEnsemble * 0.35).toInt().coerceAtLeast(10)
        val heuristicNight = (heuristicEnsemble - heuristicMorning - heuristicEvening).coerceAtLeast(5)

        // Real, independently-fitted models (Ridge / Gradient-Boosted Stumps /
        // Holt-Winters), when the per-shift dataset is available.
        val realModels = shiftRecords?.takeIf { it.isNotEmpty() }?.let { records ->
            Triple(
                PythonMLEngine.trainRidgeRegression(records, isHoliday, isExtremeWeather, isLocalEvent),
                PythonMLEngine.trainGradientBoostedStumps(records, isHoliday, isExtremeWeather, isLocalEvent),
                PythonMLEngine.trainHoltWintersSmoothing(records)
            )
        }

        val ridgeRegressionPredOut: Double
        val gradientBoostedPredOut: Double
        val holtWintersPredOut: Double
        val morningPred: Int
        val eveningPred: Int
        val nightPred: Int
        val finalEnsemble: Int
        val modelBasisNote: String
        val weeklyForecastOut: List<DayForecast>

        // Populated only on the realModels != null branch below; left at 0/40
        // (all-heuristic) otherwise, matching ForecastEnsembleResult's defaults.
        var ridgeWeightPercentOut = 0
        var gradientBoostedWeightPercentOut = 0
        var holtWintersWeightPercentOut = 0
        var heuristicWeightPercentOut = 100
        var isRealEnsembleOut = false

        if (realModels != null) {
            val (ridge, boosted, holt) = realModels

            // 60% real fitted-model consensus, 40% the instant-reacting
            // heuristic -- the real models are the more accurate long-run
            // predictors (genuinely fitted + recency-weighted), the heuristic
            // covers the gap for anomaly flags the fitted models have seen too
            // little of to have learned a reliable coefficient for. Within
            // that 60% share, each model is weighted inversely to its own
            // held-out MAE (lower error = more say), instead of a flat
            // one-third each, so a model that is genuinely more accurate on
            // this hospital's data actually gets more weight -- not just a
            // hardcoded label claiming it does.
            val ridgeMaeSafe = ridge.mae.coerceAtLeast(0.5)
            val boostedMaeSafe = boosted.mae.coerceAtLeast(0.5)
            val holtMaeSafe = holt.mae.coerceAtLeast(0.5)
            val invRidge = 1.0 / ridgeMaeSafe
            val invBoosted = 1.0 / boostedMaeSafe
            val invHolt = 1.0 / holtMaeSafe
            val invSum = invRidge + invBoosted + invHolt
            val wRidge = invRidge / invSum
            val wBoosted = invBoosted / invSum
            val wHolt = invHolt / invSum

            fun blend(real1: Int, real2: Int, real3: Int, heuristic: Int): Int {
                val weightedReal = real1 * wRidge + real2 * wBoosted + real3 * wHolt
                return (weightedReal * 0.6 + heuristic * 0.4).toInt()
            }

            morningPred = blend(ridge.nextShiftForecast.morning, boosted.nextShiftForecast.morning, holt.nextShiftForecast.morning, heuristicMorning).coerceAtLeast(10)
            eveningPred = blend(ridge.nextShiftForecast.evening, boosted.nextShiftForecast.evening, holt.nextShiftForecast.evening, heuristicEvening).coerceAtLeast(10)
            nightPred = blend(ridge.nextShiftForecast.night, boosted.nextShiftForecast.night, holt.nextShiftForecast.night, heuristicNight).coerceAtLeast(5)
            finalEnsemble = morningPred + eveningPred + nightPred

            ridgeRegressionPredOut = (ridge.nextShiftForecast.morning + ridge.nextShiftForecast.evening + ridge.nextShiftForecast.night).toDouble()
            gradientBoostedPredOut = (boosted.nextShiftForecast.morning + boosted.nextShiftForecast.evening + boosted.nextShiftForecast.night).toDouble()
            holtWintersPredOut = (holt.nextShiftForecast.morning + holt.nextShiftForecast.evening + holt.nextShiftForecast.night).toDouble()

            isRealEnsembleOut = true
            ridgeWeightPercentOut = (wRidge * 60.0).toInt()
            gradientBoostedWeightPercentOut = (wBoosted * 60.0).toInt()
            holtWintersWeightPercentOut = (wHolt * 60.0).toInt()
            heuristicWeightPercentOut = 40

            modelBasisNote = "Real Ensemble: Ridge Regression (MAE ${String.format("%.1f", ridge.mae)}, $ridgeWeightPercentOut% of blend), Gradient-Boosted Stumps (MAE ${String.format("%.1f", boosted.mae)}, $gradientBoostedWeightPercentOut% of blend), and Holt-Winters (MAE ${String.format("%.1f", holt.mae)}, $holtWintersWeightPercentOut% of blend) -- all recency-weighted and fitted on ${shiftRecords?.size ?: 0} shift records, weighted inversely to held-out MAE within their combined 60% share, blended with the instant-reacting heuristic (${heuristicWeightPercentOut}%)."

            // Per-day breakdown across the fixed roster week: each real model
            // already produced its own day-of-week-aware forecast for every
            // named day (PythonMLEngine.weekForecast), so this is the SAME
            // inverse-MAE blend used above, just repeated per day instead of
            // once for "right now" -- this is the actual fix for the roster
            // reusing one day's number across the whole week (Section 2.5 /
            // Eq. 2.10-2.11 in the report).
            weeklyForecastOut = weekDayNames.map { day ->
                val ridgeDay = ridge.weekForecast.find { it.dayName == day }
                val boostedDay = boosted.weekForecast.find { it.dayName == day }
                val holtDay = holt.weekForecast.find { it.dayName == day }
                val dayFactor = dayOfWeekFactor.getValue(day)
                val heuristicMorningDay = (heuristicMorning * dayFactor).toInt().coerceAtLeast(10)
                val heuristicEveningDay = (heuristicEvening * dayFactor).toInt().coerceAtLeast(10)
                val heuristicNightDay = (heuristicNight * dayFactor).toInt().coerceAtLeast(5)
                if (ridgeDay != null && boostedDay != null && holtDay != null) {
                    DayForecast(
                        day = day,
                        morning = blend(ridgeDay.morning, boostedDay.morning, holtDay.morning, heuristicMorningDay).coerceAtLeast(10),
                        evening = blend(ridgeDay.evening, boostedDay.evening, holtDay.evening, heuristicEveningDay).coerceAtLeast(10),
                        night = blend(ridgeDay.night, boostedDay.night, holtDay.night, heuristicNightDay).coerceAtLeast(5)
                    )
                } else {
                    // Should not happen (weekForecast always has all 7 days when
                    // shiftRecords is non-empty), but fall back to the
                    // day-adjusted heuristic alone rather than crash.
                    DayForecast(day, heuristicMorningDay, heuristicEveningDay, heuristicNightDay)
                }
            }
        } else {
            // Fallback: original heuristic-only behavior, unchanged. The three
            // *Pred values below are different arithmetic views of the same
            // heuristic formula, not independently fitted models, so no
            // per-model weight is reported (isRealEnsembleOut stays false).
            morningPred = heuristicMorning
            eveningPred = heuristicEvening
            nightPred = heuristicNight
            finalEnsemble = heuristicEnsemble
            ridgeRegressionPredOut = ridgeTrendPred
            gradientBoostedPredOut = nonLinearBoostedPred
            holtWintersPredOut = seasonalHoltWintersPred
            modelBasisNote = "Heuristic-only (per-shift dataset unavailable to this caller): single blended recency + momentum + seasonality + anomaly formula."

            // Even without the fitted models, still give each roster day its
            // own number via the empirical day-of-week factor above, instead
            // of the flat heuristic repeated seven times.
            weeklyForecastOut = weekDayNames.map { day ->
                val dayFactor = dayOfWeekFactor.getValue(day)
                DayForecast(
                    day = day,
                    morning = (heuristicMorning * dayFactor).toInt().coerceAtLeast(10),
                    evening = (heuristicEvening * dayFactor).toInt().coerceAtLeast(10),
                    night = (heuristicNight * dayFactor).toInt().coerceAtLeast(5)
                )
            }
        }

        // 5. Trend Analysis (Expert mental process of the trend)
        val momDirection = if (momentumChange >= 0) "upward" else "downward"
        val anomalyReport = mutableListOf<String>()
        if (isHoliday) anomalyReport.add("Holiday (-12% clinic dip)")
        if (isExtremeWeather) anomalyReport.add("Extreme Weather (-22% mobility dip)")
        if (isLocalEvent) anomalyReport.add("Local Event (+15% crowd surge)")
        val anomalyText = if (anomalyReport.isEmpty()) "None" else anomalyReport.joinToString(", ")

        val trendAnalysisString = "Forecasting Trend Assessment:\n" +
                "• Model Basis: $modelBasisNote\n" +
                "• Historical Dataset & Receptionist Logs: $size records analyzed. Dynamic baseline: ${String.format("%.1f", baselineAvg)} patients/day.\n" +
                "• Receptionist Real-Time Intake: Incorporating intake updates logged by Receptionist (Recent average: ${String.format("%.1f", recentAvg)}). Recency-biased starting point: ${String.format("%.1f", recencyBiasedBase)}.\n" +
                "• Shift-Wise Breakdown: Morning Shift: $morningPred, Evening Shift: $eveningPred, Night Shift: $nightPred.\n" +
                "• Non-linear Dynamics: Detected $momDirection momentum change of ${String.format("%.1f", momentumChange * 100.0)}% and a cyclical factor of ${String.format("%.2f", cyclicalFactor)}.\n" +
                "• Anomaly Adjustments: Active anomalies: [$anomalyText]. Combined multiplier: ${String.format("%.2f", anomalyMultiplier)}.\n" +
                "• Verdict: Calculated forecasted daily patient inflow of $finalEnsemble patients across Morning, Evening, and Night shifts."

        // Determine confidence level
        val confidenceLevel = when {
            isExtremeWeather -> "Low (High Volatility from Extreme Weather)"
            isHoliday && isLocalEvent -> "Low (Conflicting Anomalies)"
            isHoliday || isLocalEvent -> "Medium (Active Anomaly Factor)"
            kotlin.math.abs(momentumChange) > 0.15 -> "Medium (High Momentum Shift)"
            else -> "High (Stable Cyclical Baseline)"
        }

        return ForecastEnsembleResult(
            ridgeRegressionPred = ridgeRegressionPredOut,
            gradientBoostedPred = gradientBoostedPredOut,
            holtWintersPred = holtWintersPredOut,
            ensemblePred = finalEnsemble,
            morningPred = morningPred,
            eveningPred = eveningPred,
            nightPred = nightPred,
            fitConfidence = confidenceLevel,
            trendAnalysis = trendAnalysisString,
            isRealEnsemble = isRealEnsembleOut,
            ridgeWeightPercent = ridgeWeightPercentOut,
            gradientBoostedWeightPercent = gradientBoostedWeightPercentOut,
            holtWintersWeightPercent = holtWintersWeightPercentOut,
            heuristicWeightPercent = heuristicWeightPercentOut,
            weeklyForecast = weeklyForecastOut
        )
    }

    // --- OPTIMALITY VERIFICATION (POST-HOC HARD-CONSTRAINT AUDIT) ---
    // Independently re-checks a generated roster against every hard constraint from
    // the report's MILP formulation (Section 3.2, Constraints 3.5-3.9), rather than
    // simply trusting that the constructive roster-assignment algorithm (runConstructiveRosterAssignment
    // in MediShiftViewModel) enforced them correctly. This is deliberately a separate,
    // from-scratch scan of the persisted roster -- it does not read any solver-internal
    // state -- so it can genuinely catch a regression in the assignment algorithm.

    data class ConstraintCheckResult(
        val id: String,            // "C1".."C5", the five hard constraints, Eq. 3.5-3.9
        val name: String,         // short human-readable name
        val formula: String,      // the formal constraint, in plain-text math notation
        val checksPerformed: Int, // number of individual (person[, day/window]) instances checked
        val checksSatisfied: Int,
        val violations: List<String> // human-readable description of each failing instance
    ) {
        val checksViolated: Int get() = checksPerformed - checksSatisfied
        val isFullySatisfied: Boolean get() = checksViolated == 0
    }

    data class OptimalityVerificationReport(
        val constraints: List<ConstraintCheckResult>,
        val totalChecks: Int,
        val totalSatisfied: Int,
        val overallPercent: Double,
        val allConstraintsSatisfied: Boolean,
        val rosterShiftCount: Int,
        val staffAudited: Int
    )

    private val rosterWeekDays = listOf("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

    /**
     * Re-derives, from the persisted roster alone, whether every instance of
     * Constraints 3.5-3.9 actually holds -- these five are the ONLY hard
     * constraints in the report's MILP formulation, enforced exactly and
     * never traded away:
     *  C1 (Eq. 3.5): sum_s x(i,d,s) <= 1               -- at most one shift per person per day
     *  C2 (Eq. 3.6): sum_(d,s) x(i,d,s) <= 5            -- at most five shifts per person per week
     *  C3 (Eq. 3.7): x(i,d,s) <= 1 - L(i,d)             -- nobody works on an approved-leave day
     *  C4 (Eq. 3.8): x(i,d,N)+x(i,d+1,M)+x(i,d+1,E) <= 1-- a Night shift blocks Morning/Evening the next day
     *  C5 (Eq. 3.9): sum_(k=0..3) x(i,d+k,N) <= 3       -- no four consecutive Night shifts, checked over
     *                                                       every rolling four-day window of the week
     *
     * `staffPool` should be the full active optimization-pool staff list (I in the
     * report's notation), not just staff who happen to appear in the roster, so C2 and
     * C5 are checked for every person the constraint actually quantifies over.
     */
    fun verifyMilpConstraints(
        rosterItems: List<FinalRosterItem>,
        staffPool: List<StaffProfile>,
        leaveRequests: List<LeaveRequest>
    ): OptimalityVerificationReport {
        val days = rosterWeekDays
        val nameById = rosterItems.associate { it.staffId to it.staffName } + staffPool.associate { it.id to it.name }
        val poolIds = staffPool.map { it.id }.distinct()
        val byStaffDay = rosterItems.groupBy { it.staffId to it.date.trim() }

        fun nameFor(staffId: Int) = nameById[staffId] ?: "Staff #$staffId"

        // C1: one shift per person per day.
        val c1Violations = mutableListOf<String>()
        var c1Satisfied = 0
        byStaffDay.forEach { (key, items) ->
            if (items.size <= 1) {
                c1Satisfied++
            } else {
                c1Violations.add("${nameFor(key.first)} was assigned ${items.size} shifts on ${key.second} (${items.joinToString { it.shiftSlot }})")
            }
        }
        val c1Checks = byStaffDay.size

        // C2: weekly shift cap, checked for every active pool member.
        val c2Violations = mutableListOf<String>()
        var c2Satisfied = 0
        poolIds.forEach { staffId ->
            val count = rosterItems.count { it.staffId == staffId }
            if (count <= 5) {
                c2Satisfied++
            } else {
                c2Violations.add("${nameFor(staffId)} is assigned $count shifts this week (limit 5)")
            }
        }
        val c2Checks = poolIds.size

        // C3: approved leave always respected.
        val c3Violations = mutableListOf<String>()
        var c3Satisfied = 0
        rosterItems.forEach { item ->
            val onLeave = leaveRequests.any { lr ->
                lr.staffId == item.staffId && lr.status == "Approved" &&
                    lr.days.split(",").any { it.trim().equals(item.date.trim(), ignoreCase = true) }
            }
            if (!onLeave) {
                c3Satisfied++
            } else {
                c3Violations.add("${item.staffName} was rostered for ${item.shiftSlot} on ${item.date} despite approved leave that day")
            }
        }
        val c3Checks = rosterItems.size

        // C4: rest the day immediately after a Night shift.
        val c4Violations = mutableListOf<String>()
        var c4Checks = 0
        var c4Satisfied = 0
        poolIds.forEach { staffId ->
            for (d in 0..5) { // day d and d+1 both valid for a 7-day week -> d in D \ {7}
                val today = days[d]
                val tomorrow = days[d + 1]
                val workedNightToday = (byStaffDay[staffId to today] ?: emptyList()).any { it.shiftSlot == "Night" }
                if (!workedNightToday) continue
                c4Checks++
                val tomorrowShifts = (byStaffDay[staffId to tomorrow] ?: emptyList()).map { it.shiftSlot }
                val clashShift = tomorrowShifts.firstOrNull { it == "Morning" || it == "Evening" }
                if (clashShift == null) {
                    c4Satisfied++
                } else {
                    c4Violations.add("${nameFor(staffId)} worked Night on $today then $clashShift on $tomorrow")
                }
            }
        }

        // C5: no four consecutive Night shifts, over every rolling four-day window.
        val c5Violations = mutableListOf<String>()
        var c5Satisfied = 0
        poolIds.forEach { staffId ->
            for (d in 0..3) { // windows {d..d+3} starting Mon..Thu so d+3 stays within the week
                val windowDays = (d..d + 3).map { days[it] }
                val nightsInWindow = windowDays.count { day ->
                    (byStaffDay[staffId to day] ?: emptyList()).any { it.shiftSlot == "Night" }
                }
                if (nightsInWindow <= 3) {
                    c5Satisfied++
                } else {
                    c5Violations.add("${nameFor(staffId)} worked Night on all four days of ${windowDays.first()}-${windowDays.last()}")
                }
            }
        }
        val c5Checks = poolIds.size * 4

        val constraints = listOf(
            ConstraintCheckResult("C1", "One shift per person per day", "Σs x(i,d,s) ≤ 1", c1Checks, c1Satisfied, c1Violations),
            ConstraintCheckResult("C2", "Weekly shift cap (5)", "Σ(d,s) x(i,d,s) ≤ 5", c2Checks, c2Satisfied, c2Violations),
            ConstraintCheckResult("C3", "Approved leave respected", "x(i,d,s) ≤ 1 − L(i,d)", c3Checks, c3Satisfied, c3Violations),
            ConstraintCheckResult("C4", "Rest after a Night shift", "x(i,d,N)+x(i,d+1,M)+x(i,d+1,E) ≤ 1", c4Checks, c4Satisfied, c4Violations),
            ConstraintCheckResult("C5", "Max 3 consecutive Night shifts", "Σ(k=0..3) x(i,d+k,N) ≤ 3", c5Checks, c5Satisfied, c5Violations)
        )

        val totalChecks = constraints.sumOf { it.checksPerformed }
        val totalSatisfied = constraints.sumOf { it.checksSatisfied }

        return OptimalityVerificationReport(
            constraints = constraints,
            totalChecks = totalChecks,
            totalSatisfied = totalSatisfied,
            overallPercent = if (totalChecks == 0) 100.0 else (totalSatisfied.toDouble() / totalChecks) * 100.0,
            allConstraintsSatisfied = constraints.all { it.isFullySatisfied },
            rosterShiftCount = rosterItems.size,
            staffAudited = poolIds.size
        )
    }
}
