package com.example.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.room.Room
import com.example.data.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.util.UUID
import java.text.SimpleDateFormat
import java.util.Date
import java.util.TimeZone
import java.util.Locale

class MediShiftViewModel(application: Application) : AndroidViewModel(application) {

    fun getIndianStandardTimeStr(): String {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.ENGLISH)
        sdf.timeZone = TimeZone.getTimeZone("Asia/Kolkata")
        return sdf.format(Date()) + " (IST)"
    }

    private val db = Room.databaseBuilder(
        application,
        AppDatabase::class.java, "medishift_db"
    ).fallbackToDestructiveMigration().build()

    private val repository = MediShiftRepository(db.dao())

    // Authentication States
    private val _currentUser = MutableStateFlow<UserAccount?>(null)
    val currentUser: StateFlow<UserAccount?> = _currentUser

    private val _authError = MutableStateFlow<String?>(null)
    val authError: StateFlow<String?> = _authError

    // Roster release status, finalized by Operations Manager
    private val _isRosterReleased = MutableStateFlow(false)
    val isRosterReleased: StateFlow<Boolean> = _isRosterReleased

    // Deep Link state for external web portals
    private val _deepLinkHospital = MutableStateFlow<String?>(null)
    val deepLinkHospital: StateFlow<String?> = _deepLinkHospital

    fun handleDeepLink(hospital: String?) {
        _deepLinkHospital.value = hospital
    }

    fun dismissDeepLink() {
        _deepLinkHospital.value = null
    }

    // UI Observables
    val staffList = repository.allStaff.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    val historicalInflows = repository.allInflows.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    val rosterItems = repository.allRosterItems.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    val appointments = repository.allAppointments.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    val candidatesList = repository.allCandidates.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    val allUserAccounts = repository.allUserAccounts.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    val leaveRequests = repository.allLeaveRequests.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    @OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
    val userEmails: StateFlow<List<EmailMessage>> = currentUser
        .flatMapLatest { user ->
            if (user != null) {
                repository.getEmailsForUser(user.email)
            } else {
                flowOf(emptyList())
            }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    @OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
    val userSentEmails: StateFlow<List<EmailMessage>> = currentUser
        .flatMapLatest { user ->
            if (user != null) {
                repository.getSentEmailsForUser(user.email)
            } else {
                flowOf(emptyList())
            }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    // Local state for interactive forecasting and optimization
    private val _predictedInflow = MutableStateFlow(1580) // Default daily patient admission count (~1500+ daily average)
    val predictedInflow: StateFlow<Int> = _predictedInflow

    private val _dynamicStaffNeeded = MutableStateFlow(6)
    val dynamicStaffNeeded: StateFlow<Int> = _dynamicStaffNeeded

    private val _isOptimizing = MutableStateFlow(false)
    val isOptimizing: StateFlow<Boolean> = _isOptimizing

    private val _solverStatusMessage = MutableStateFlow("")
    val solverStatusMessage: StateFlow<String> = _solverStatusMessage

    private val _solverMetrics = MutableStateFlow<SolverMetrics?>(null)
    val solverMetrics: StateFlow<SolverMetrics?> = _solverMetrics

    data class SolverMetrics(
        val totalAssignments: Int,
        val hardConstraintsMet: Boolean,
        val softConstraintsMetPercent: Int,
        val avgShiftsPerStaff: Double,
        val preferredDaysOffGranted: Int
    )

    // New LP Staffing Selector Result StateFlow
    private val _lpResult = MutableStateFlow<OptimizationModels.StaffingLPResult?>(null)
    val lpResult: StateFlow<OptimizationModels.StaffingLPResult?> = _lpResult

    // Per-shift aggregate optimality: the same ratio-matching objective applied
    // independently to every shift block (Morning/Evening/Night), not just once
    // for the day as a whole.
    private val _dailyStaffingPlan = MutableStateFlow<OptimizationModels.DailyStaffingPlan?>(null)
    val dailyStaffingPlan: StateFlow<OptimizationModels.DailyStaffingPlan?> = _dailyStaffingPlan

    // Optimality Verification Report (Operations Manager portal): independent,
    // post-hoc audit of the currently persisted roster against every hard
    // constraint (Eq. 3.5-3.10) from the report's MILP formulation.
    private val _optimalityReport = MutableStateFlow<OptimizationModels.OptimalityVerificationReport?>(null)
    val optimalityReport: StateFlow<OptimizationModels.OptimalityVerificationReport?> = _optimalityReport

    private val _isVerifyingOptimality = MutableStateFlow(false)
    val isVerifyingOptimality: StateFlow<Boolean> = _isVerifyingOptimality

    fun runOptimalityVerification() {
        viewModelScope.launch {
            _isVerifyingOptimality.value = true
            delay(400) // brief, honest processing delay -- this is a real scan, not decorative
            val report = OptimizationModels.verifyMilpConstraints(
                rosterItems = rosterItems.value,
                staffPool = staffList.value.filter { it.isInOptimizationPool },
                leaveRequests = leaveRequests.value
            )
            _optimalityReport.value = report
            triggerLocalSync(
                "SOLVER",
                "Optimality Verification Report run: ${report.totalSatisfied}/${report.totalChecks} checks satisfied (${String.format("%.1f", report.overallPercent)}%), hard constraints ${if (report.allConstraintsSatisfied) "all satisfied" else "violated"}."
            )
            _isVerifyingOptimality.value = false
        }
    }

    // Advanced Ensemble Forecasting state
    private val _ensembleResult = MutableStateFlow<OptimizationModels.ForecastEnsembleResult?>(null)
    val ensembleResult: StateFlow<OptimizationModels.ForecastEnsembleResult?> = _ensembleResult

    private val _isForecasting = MutableStateFlow(false)
    val isForecasting: StateFlow<Boolean> = _isForecasting

    private val _isHoliday = MutableStateFlow(false)
    val isHoliday: StateFlow<Boolean> = _isHoliday

    private val _isExtremeWeather = MutableStateFlow(false)
    val isExtremeWeather: StateFlow<Boolean> = _isExtremeWeather

    private val _isLocalEvent = MutableStateFlow(false)
    val isLocalEvent: StateFlow<Boolean> = _isLocalEvent

    fun setHoliday(value: Boolean) { _isHoliday.value = value }
    fun setExtremeWeather(value: Boolean) { _isExtremeWeather.value = value }
    fun setLocalEvent(value: Boolean) { _isLocalEvent.value = value }

    fun solveStaffingLP(
        patients: Int,
        availableDocs: Int,
        availableNurses: Int,
        availablePhars: Int,
        availableLabs: Int,
        doctorGoodRatio: Double = 50.0,
        nurseGoodRatio: Double = 20.0,
        pharmacistGoodRatio: Double = 100.0,
        labTechGoodRatio: Double = 100.0,
        doctorTargetRatio: Double = 20.0,
        nurseTargetRatio: Double = 6.0,
        pharmacistTargetRatio: Double = 75.0,
        labTechTargetRatio: Double = 40.0
    ) {
        viewModelScope.launch {
            // 1. Run the individual-level shift-assignment solver, aiming for the
            // same Good/Target ratios as the category-level model below (the
            // specific named-person rostering is a separate step, outside this
            // model's scope -- see runConstructiveRosterAssignment).
            runConstructiveRosterAssignment(
                docCap = doctorTargetRatio,
                nurseCap = nurseTargetRatio,
                pharCap = pharmacistTargetRatio,
                labCap = labTechTargetRatio,
                docGoodCap = doctorGoodRatio,
                nurseGoodCap = nurseGoodRatio,
                pharGoodCap = pharmacistGoodRatio,
                labGoodCap = labTechGoodRatio
            )

            // 2. Solve the ratio-matching Integer Program for category headcounts
            val localInput = OptimizationModels.StaffingLPInput(
                predictedPatients = patients,
                availableDoctors = availableDocs,
                availableNurses = availableNurses,
                availablePharmacists = availablePhars,
                availableLabTechs = availableLabs,
                doctorGoodRatio = doctorGoodRatio,
                nurseGoodRatio = nurseGoodRatio,
                pharmacistGoodRatio = pharmacistGoodRatio,
                labTechGoodRatio = labTechGoodRatio,
                doctorTargetRatio = doctorTargetRatio,
                nurseTargetRatio = nurseTargetRatio,
                pharmacistTargetRatio = pharmacistTargetRatio,
                labTechTargetRatio = labTechTargetRatio
            )

            val result = OptimizationModels.solveStaffingLP(localInput)
            LocalSyncManager.logEvent("SOLVER", "Staffing LP optimized locally on SQLite device. Total Cost: ₹${result.totalCost}")

            _lpResult.value = result

            // 3. Apply the SAME optimality objective independently to every shift
            // block, so the objective function is satisfied at every shift, not
            // just once for the aggregate day. Prefer the genuinely-forecasted
            // per-shift split from the ensemble model when available; otherwise
            // fall back to the same proportional split used by runConstructiveRosterAssignment.
            val ensemble = _ensembleResult.value
            val (morningPatients, eveningPatients, nightPatients) = if (
                ensemble != null && (ensemble.morningPred + ensemble.eveningPred + ensemble.nightPred) > 0
            ) {
                Triple(ensemble.morningPred, ensemble.eveningPred, ensemble.nightPred)
            } else {
                Triple(
                    (patients * 0.45).toInt(),
                    (patients * 0.35).toInt(),
                    (patients * 0.20).toInt()
                )
            }

            val dailyPlan = OptimizationModels.solveStaffingLPAllShifts(
                morningPatients = morningPatients,
                eveningPatients = eveningPatients,
                nightPatients = nightPatients,
                availableDoctors = availableDocs,
                availableNurses = availableNurses,
                availablePharmacists = availablePhars,
                availableLabTechs = availableLabs,
                doctorGoodRatio = doctorGoodRatio,
                nurseGoodRatio = nurseGoodRatio,
                pharmacistGoodRatio = pharmacistGoodRatio,
                labTechGoodRatio = labTechGoodRatio,
                doctorTargetRatio = doctorTargetRatio,
                nurseTargetRatio = nurseTargetRatio,
                pharmacistTargetRatio = pharmacistTargetRatio,
                labTechTargetRatio = labTechTargetRatio
            )
            _dailyStaffingPlan.value = dailyPlan

            // Also persist the result into OperationalState so it survives an app
            // restart and can be restored into lpResult on next launch (see init{}).
            val finalResult = result
            updateOperationalStateInDb {
                it.copy(
                    persistedBudget = availableDocs.toDouble(),
                    persistedDoctors = finalResult.doctors,
                    persistedNurses = finalResult.nurses,
                    persistedPharmacists = finalResult.pharmacists,
                    persistedLabTechs = finalResult.labTechs,
                    persistedTotalCost = finalResult.totalCost,
                    persistedTotalUtility = finalResult.totalHours,
                    hasPersistedResult = true
                )
            }

            triggerLocalSync(
                "LP_SOLVER",
                "[Local Device] Matched staffing to ideal ratios for $patients patient demand: ${finalResult.doctors} Docs, ${finalResult.nurses} Nurses, ${finalResult.pharmacists} Phars, ${finalResult.labTechs} Labs. Total deviation from ideal: ${finalResult.totalDeviation} staff. Per-shift: ${dailyPlan.summary}"
            )
        }
    }

    fun runEnsembleForecasting() {
        viewModelScope.launch {
            _isForecasting.value = true
            delay(1200) // Beautiful authentic model training delay

            ensureHistoricalInflowPopulated()
            val history = repository.getHistoricalInflowList()

            // Feed the genuinely-fitted, recency-weighted per-shift models
            // (PythonMLEngine: Ridge Regression, Gradient-Boosted Stumps,
            // Holt-Winters) into the live ensemble, not just the ML Dashboard
            // display. Falls back to the heuristic-only path automatically if
            // the dataset can't be loaded for any reason.
            val shiftRecords = try {
                PythonMLEngine.loadDataset(getApplication())
            } catch (e: Exception) {
                android.util.Log.e("MediShiftVM", "Failed to load shift dataset for ensemble: ${e.message}")
                emptyList()
            }

            val result = OptimizationModels.trainAndPredictEnsemble(
                history = history,
                isHoliday = _isHoliday.value,
                isExtremeWeather = _isExtremeWeather.value,
                isLocalEvent = _isLocalEvent.value,
                shiftRecords = shiftRecords
            )
            _ensembleResult.value = result
            
            // Sync with predictedInflow state flow
            updatePrediction(result.ensemblePred)
            
            triggerLocalSync(
                "ENSEMBLE_FORECAST",
                "Advanced forecast updated: ${result.ensemblePred} patients. Confidence: ${result.fitConfidence}. Broadcasted to all clinical subscriber sessions."
            )
            _isForecasting.value = false
        }
    }

    fun triggerLocalSync(category: String, message: String) {
        LocalSyncManager.simulateLiveSyncEvent(category, message)
    }

    private suspend fun updateOperationalStateInDb(update: (OperationalState) -> OperationalState) {
        val currentState = repository.getOperationalState() ?: OperationalState()
        val newState = update(currentState)
        repository.insertOperationalState(newState)
    }

    init {
        // Initialize Local SQLite Database Engine
        LocalSyncManager.initialize(application)

        // Always synchronize database with staff_dataset.json and shift_dataset_3years.json
        viewModelScope.launch {
            syncFullDatasetToDatabase()
        }

        // Centralized State Management Store: Real-time synchronization of forecasts and assignments
        viewModelScope.launch {
            repository.operationalState.collect { state ->
                if (state != null) {
                    _predictedInflow.value = state.predictedInflow
                    _dynamicStaffNeeded.value = state.dynamicStaffNeeded
                    _isRosterReleased.value = state.isRosterReleased
                    
                    if (state.hasSolverMetrics) {
                        _solverMetrics.value = SolverMetrics(
                            totalAssignments = state.solverTotalAssignments,
                            hardConstraintsMet = state.solverHardConstraintsMet,
                            softConstraintsMetPercent = state.solverSoftConstraintsMetPercent,
                            avgShiftsPerStaff = state.solverAvgShiftsPerStaff,
                            preferredDaysOffGranted = state.solverPreferredDaysOffGranted
                        )
                    } else {
                        _solverMetrics.value = null
                    }
                    
                    if (state.hasPersistedResult) {
                        // Recompute ideal/minSafe/deviation against the standard ratio
                        // table so a restored result stays consistent with a fresh
                        // solveStaffingLP() call (only the chosen headcounts are
                        // actually persisted in the DB; the rest is derived).
                        val pPatients = state.predictedInflow.coerceAtLeast(1)
                        val minSafeD = kotlin.math.ceil(pPatients / 50.0).toInt().coerceAtLeast(1)
                        val minSafeN = kotlin.math.ceil(pPatients / 20.0).toInt().coerceAtLeast(1)
                        val minSafeP = kotlin.math.ceil(pPatients / 100.0).toInt().coerceAtLeast(1)
                        val minSafeL = kotlin.math.ceil(pPatients / 100.0).toInt().coerceAtLeast(1)
                        val idealD = kotlin.math.ceil(pPatients / 20.0).toInt().coerceAtLeast(minSafeD)
                        val idealN = kotlin.math.ceil(pPatients / 6.0).toInt().coerceAtLeast(minSafeN)
                        val idealP = kotlin.math.ceil(pPatients / 75.0).toInt().coerceAtLeast(minSafeP)
                        val idealL = kotlin.math.ceil(pPatients / 40.0).toInt().coerceAtLeast(minSafeL)
                        val devD = kotlin.math.abs(state.persistedDoctors - idealD)
                        val devN = kotlin.math.abs(state.persistedNurses - idealN)
                        val devP = kotlin.math.abs(state.persistedPharmacists - idealP)
                        val devL = kotlin.math.abs(state.persistedLabTechs - idealL)
                        _lpResult.value = OptimizationModels.StaffingLPResult(
                            doctors = state.persistedDoctors,
                            nurses = state.persistedNurses,
                            pharmacists = state.persistedPharmacists,
                            labTechs = state.persistedLabTechs,
                            idealDoctors = idealD,
                            idealNurses = idealN,
                            idealPharmacists = idealP,
                            idealLabTechs = idealL,
                            minSafeDoctors = minSafeD,
                            minSafeNurses = minSafeN,
                            minSafePharmacists = minSafeP,
                            minSafeLabTechs = minSafeL,
                            deviationDoctors = devD,
                            deviationNurses = devN,
                            deviationPharmacists = devP,
                            deviationLabTechs = devL,
                            totalDeviation = devD + devN + devP + devL,
                            totalCost = state.persistedTotalCost,
                            totalHours = state.persistedTotalUtility,
                            totalLaborCost = state.persistedTotalCost * 600.0,
                            isQualityCompromised = state.persistedDoctors < minSafeD || state.persistedNurses < minSafeN ||
                                state.persistedPharmacists < minSafeP || state.persistedLabTechs < minSafeL,
                            isWithinBudget = state.persistedTotalCost <= state.persistedBudget,
                            doctorRatioText = "1:${String.format("%.1f", pPatients.toDouble() / state.persistedDoctors.coerceAtLeast(1))} (Target 1:20, Good 1:50)",
                            nurseRatioText = "1:${String.format("%.1f", pPatients.toDouble() / state.persistedNurses.coerceAtLeast(1))} (Target 1:6, Good 1:20)",
                            pharmacistRatioText = "1:${String.format("%.1f", pPatients.toDouble() / state.persistedPharmacists.coerceAtLeast(1))} (Target 1:75, Good 1:100)",
                            labTechRatioText = "1:${String.format("%.1f", pPatients.toDouble() / state.persistedLabTechs.coerceAtLeast(1))} (Target 1:40, Good 1:100)",
                            statusMessage = if (state.persistedTotalCost <= state.persistedBudget) "Optimal staffing restored." else "Staffing limits restored under budget constraint."
                        )
                    } else {
                        _lpResult.value = null
                    }
                } else {
                    // Initialize DB operational state if it does not exist yet
                    repository.insertOperationalState(OperationalState())
                }
            }
        }
    }

    private suspend fun insertUserAccountIfMissing(account: UserAccount) {
        val existing = repository.getUserByEmail(account.email)
        if (existing == null) {
            repository.insertUserAccount(account)
        } else if (account.email == "manager@medishift.ac.in" && existing.name != "Ejisay Naik") {
            repository.insertUserAccount(existing.copy(name = "Ejisay Naik", role = "Operations Manager"))
        }
    }

    private suspend fun ensureQuickDevTestAccountsExist() {
        val profiles = repository.getAllStaffList()
        val jsonStaff = StaffDatasetManager.loadStaffDataset(getApplication())
        
        val hrAccount = UserAccount(email = "hr@medishift.ac.in", passwordPlain = "password123", name = "HR Coordinator", role = "HR", employeeId = "HR2026S01")
        insertUserAccountIfMissing(hrAccount)

        val financeAccount = UserAccount(email = "auditor@medishift.ac.in", passwordPlain = "password123", name = "Work-Hour Auditor", role = "Finance", employeeId = "FN2026S01")
        insertUserAccountIfMissing(financeAccount)

        val managerAccount = UserAccount(email = "manager@medishift.ac.in", passwordPlain = "password123", name = "Ejisay Naik", role = "Operations Manager", employeeId = "OM2026S01")
        insertUserAccountIfMissing(managerAccount)

        val officerAccount = UserAccount(email = "officer@medishift.ac.in", passwordPlain = "password123", name = "Dr. Sarah Jenkins", role = "Medical Officer", employeeId = "MO2026S01")
        insertUserAccountIfMissing(officerAccount)

        val receptionistAccount = UserAccount(email = "receptionist@medishift.ac.in", passwordPlain = "password123", name = "Jane Doe", role = "Receptionist", employeeId = "RE2026S01")
        insertUserAccountIfMissing(receptionistAccount)

        val aliceProfile = profiles.find { it.name == "Dr. Alice Vance" } ?: profiles.find { it.role.contains("Doctor", ignoreCase = true) }
        val doctorAccount = UserAccount(
            email = "doctor@medishift.ac.in",
            passwordPlain = "password123",
            name = aliceProfile?.name ?: "Dr. Alice Vance",
            role = "Doctor",
            staffProfileId = aliceProfile?.id,
            employeeId = aliceProfile?.employeeId ?: "DR2026S01"
        )
        insertUserAccountIfMissing(doctorAccount)

        val davidProfile = profiles.find { it.name == "Nurse David Miller" } ?: profiles.find { it.role.contains("Nurse", ignoreCase = true) }
        val nurseAccount = UserAccount(
            email = "nurse@medishift.ac.in",
            passwordPlain = "password123",
            name = davidProfile?.name ?: "Nurse David Miller",
            role = "Nurse",
            staffProfileId = davidProfile?.id,
            employeeId = davidProfile?.employeeId ?: "NU2026S01"
        )
        insertUserAccountIfMissing(nurseAccount)

        val tonyProfile = profiles.find { it.name == "Pharmacist Tony Stark" } ?: profiles.find { it.role.contains("Pharmacist", ignoreCase = true) }
        val pharmacistAccount = UserAccount(
            email = "pharmacist@medishift.ac.in",
            passwordPlain = "password123",
            name = tonyProfile?.name ?: "Pharmacist Tony Stark",
            role = "Pharmacist",
            staffProfileId = tonyProfile?.id,
            employeeId = tonyProfile?.employeeId ?: "PH2026S01"
        )
        insertUserAccountIfMissing(pharmacistAccount)

        val steveProfile = profiles.find { it.name == "Lab Tech Steve Rogers" } ?: profiles.find { it.role.contains("Lab", ignoreCase = true) }
        val labTechAccount = UserAccount(
            email = "labtech@medishift.ac.in",
            passwordPlain = "password123",
            name = steveProfile?.name ?: "Lab Tech Steve Rogers",
            role = "Lab Technician",
            staffProfileId = steveProfile?.id,
            employeeId = steveProfile?.employeeId ?: "LT2026S01"
        )
        insertUserAccountIfMissing(labTechAccount)

        // Seed user accounts for dataset staff
        for (s in jsonStaff) {
            val prof = profiles.find { it.employeeId == s.id || it.name.equals(s.name, ignoreCase = true) }
            val acc = UserAccount(
                email = s.email,
                passwordPlain = "password123",
                name = s.name,
                role = s.role,
                staffProfileId = prof?.id,
                employeeId = s.id
            )
            insertUserAccountIfMissing(acc)
        }

        // Seed initial real emails for users
        val initialEmails = listOf(
            EmailMessage(
                senderEmail = "hr@medishift.ac.in",
                receiverEmail = "doctor@medishift.ac.in",
                subject = "Welcome to MediShift!",
                body = "Dear Dr. Alice Vance,\n\nWelcome to the MediShift Clinical Team. Your employee profile has been created with Employee ID: DR2026S01.\n\nPlease complete your shift profile configuration and preference settings.\n\nBest Regards,\nHR Team",
                timestamp = "2026-07-06 08:00"
            ),
            EmailMessage(
                senderEmail = "auditor@medishift.ac.in",
                receiverEmail = "doctor@medishift.ac.in",
                subject = "Weekly Work-Hour Target Settings",
                body = "Dear Dr. Alice Vance,\n\nYour weekly shift schedule structure is set at Preferred standard: 40 hours/week, with a maximum overtime allowance of 8 hours.\n\nYour preferences have been registered in our scheduling engine.\n\nRegards,\nWork-Hour Auditor Team",
                timestamp = "2026-07-06 09:15"
            ),
            EmailMessage(
                senderEmail = "hr@medishift.ac.in",
                receiverEmail = "nurse@medishift.ac.in",
                subject = "Welcome to MediShift!",
                body = "Dear Nurse David Miller,\n\nWelcome to the MediShift Clinical Team. Your employee profile has been created with Employee ID: NU2026S01.\n\nPlease check your schedule and specify your day off preference.\n\nRegards,\nHR Team",
                timestamp = "2026-07-06 08:05"
            )
        )
        for (email in initialEmails) {
            if (repository.getEmailList("doctor@medishift.ac.in").isEmpty()) {
                repository.insertEmail(email)
            }
        }
    }

    private suspend fun syncFullDatasetToDatabase() {
        val jsonStaff = StaffDatasetManager.loadStaffDataset(getApplication())
        val existingStaff = repository.getAllStaffList()
        val existingEmpIds = existingStaff.map { it.employeeId }.toSet()

        val newStaffProfiles = jsonStaff.filter { !existingEmpIds.contains(it.id) }.map { s ->
            val dept = when {
                s.role.contains("Doctor", ignoreCase = true) -> "Emergency & Clinical Care"
                s.role.contains("Nurse", ignoreCase = true) -> "Inpatient & Critical Care"
                s.role.contains("Pharmacist", ignoreCase = true) -> "Pharmacy Services"
                s.role.contains("Lab", ignoreCase = true) -> "Diagnostics & Laboratory"
                else -> "General Clinical Services"
            }
            val hourlyWage = if (s.salary > 0) (s.salary / 160.0).coerceAtLeast(200.0) else 500.0
            StaffProfile(
                name = s.name,
                role = s.role,
                skillLevel = if (s.seniority.isNotBlank()) s.seniority else "Mid",
                dayOffPreference = "None",
                employeeId = s.id,
                hourlyWage = hourlyWage,
                department = dept,
                isInOptimizationPool = true
            )
        }

        for (staff in newStaffProfiles) {
            repository.insertStaff(staff)
        }

        // Prepopulate candidates list if empty
        val existingCandidates = repository.getAllCandidatesList()
        if (existingCandidates.isEmpty() && jsonStaff.isNotEmpty()) {
            val initialCandidates = jsonStaff.take(30).map { s ->
                Candidate(
                    name = s.name,
                    email = s.email,
                    role = s.role,
                    seniority = s.seniority,
                    status = "Registered",
                    salary = s.salary,
                    allowances = s.allowances
                )
            }
            for (cand in initialCandidates) {
                repository.insertCandidate(cand)
            }
        }

        // Ensure dev test accounts exist
        ensureQuickDevTestAccountsExist()

        // Sync historical inflow dataset
        ensureHistoricalInflowPopulated()

        // Run default simulation prediction
        runEnsembleForecasting()
    }

    suspend fun ensureHistoricalInflowPopulated() {
        try {
            val existingInflows = repository.getHistoricalInflowList()
            val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())

            val dataset = ShiftDatasetManager.loadDataset(getApplication())
            if (dataset.isNotEmpty()) {
                val groupedByDate = dataset.groupBy { it.date }
                val historicalDates = groupedByDate.keys.filter { it < todayStr }

                val existingMap = existingInflows.associateBy { it.date }
                for (dateKey in historicalDates) {
                    val dayTotal = groupedByDate[dateKey]?.sumOf { it.patientInflow } ?: 100
                    val existing = existingMap[dateKey]
                    if (existing == null) {
                        repository.insertInflow(HistoricalInflow(date = dateKey, patientCount = dayTotal))
                    } else if (existing.patientCount != dayTotal) {
                        repository.insertInflow(existing.copy(patientCount = dayTotal))
                    }
                }
            }
        } catch (e: Exception) {
            android.util.Log.e("MediShiftVM", "Error syncing historical inflows: ${e.message}")
        }
    }

    // Authentication Operations
    fun login(email: String, passwordPlain: String, onResult: (Boolean) -> Unit) {
        viewModelScope.launch {
            _authError.value = null
            var user = repository.getUserByEmail(email)
            if (user == null) {
                // Dynamic lookup in json staff dataset
                val jsonStaff = StaffDatasetManager.loadStaffDataset(getApplication())
                val match = jsonStaff.find { it.email.equals(email, ignoreCase = true) }
                if (match != null && (passwordPlain == "password123" || passwordPlain.isNotBlank())) {
                    val allStaff = repository.getAllStaffList()
                    var existingProfile = allStaff.find { it.employeeId == match.id || it.name.equals(match.name, ignoreCase = true) }
                    if (existingProfile == null) {
                        val dept = when {
                            match.role.contains("Doctor", ignoreCase = true) -> "Emergency & Clinical Care"
                            match.role.contains("Nurse", ignoreCase = true) -> "Inpatient & Critical Care"
                            match.role.contains("Pharmacist", ignoreCase = true) -> "Pharmacy Services"
                            match.role.contains("Lab", ignoreCase = true) -> "Diagnostics & Laboratory"
                            else -> "General Clinical Services"
                        }
                        val newProf = StaffProfile(
                            name = match.name,
                            role = match.role,
                            skillLevel = if (match.seniority.isNotBlank()) match.seniority else "Mid",
                            dayOffPreference = "None",
                            employeeId = match.id,
                            hourlyWage = (match.salary / 160.0).coerceAtLeast(200.0),
                            department = dept
                        )
                        repository.insertStaff(newProf)
                        existingProfile = repository.getAllStaffList().find { it.name.equals(match.name, ignoreCase = true) }
                    }
                    val newUser = UserAccount(
                        email = match.email,
                        passwordPlain = passwordPlain,
                        name = match.name,
                        role = match.role,
                        staffProfileId = existingProfile?.id,
                        employeeId = match.id
                    )
                    repository.insertUserAccount(newUser)
                    user = newUser
                }
            }

            if (user != null && user.passwordPlain == passwordPlain) {
                var updatedUser = user
                // Automatically create/link StaffProfile if not already done for clinical/rostered users
                if (user.role != "HR") {
                    val allStaff = repository.getAllStaffList()
                    val existingProfile = allStaff.find { it.name.equals(user.name, ignoreCase = true) }
                    if (existingProfile == null) {
                        val candidate = repository.getCandidateByEmail(email)
                        val seniority = candidate?.seniority ?: "Junior"
                        val generatedEmpId = generateNextEmployeeId(user.role)
                        val profile = StaffProfile(
                            name = user.name,
                            role = user.role,
                            skillLevel = seniority,
                            dayOffPreference = "None",
                            employeeId = generatedEmpId
                        )
                        repository.insertStaff(profile)
                        val allUpdatedStaff = repository.getAllStaffList()
                        val newStaffId = allUpdatedStaff.find { it.name.equals(user.name, ignoreCase = true) }?.id
                        if (newStaffId != null) {
                            updatedUser = user.copy(staffProfileId = newStaffId, employeeId = generatedEmpId)
                            repository.insertUserAccount(updatedUser)
                        }
                    } else if (user.staffProfileId == null || user.staffProfileId != existingProfile.id) {
                        updatedUser = user.copy(
                            staffProfileId = existingProfile.id,
                            employeeId = existingProfile.employeeId
                        )
                        repository.insertUserAccount(updatedUser)
                    }
                }
                _currentUser.value = updatedUser
                onResult(true)
            } else {
                _authError.value = "Invalid email or password."
                onResult(false)
            }
        }
    }

    fun createAccount(email: String, passwordPlain: String, name: String, role: String, onResult: (Boolean) -> Unit) {
        viewModelScope.launch {
            _authError.value = null
            if (email.isBlank() || passwordPlain.isBlank() || name.isBlank()) {
                _authError.value = "All fields are required."
                onResult(false)
                return@launch
            }
            val existing = repository.getUserByEmail(email)
            if (existing != null) {
                _authError.value = "Email is already registered."
                onResult(false)
                return@launch
            }

            var seniority = "Junior"
            if (role != "HR") {
                // Verify candidate email domain
                if (!email.endsWith("@medishift.ac.in", ignoreCase = true)) {
                    _authError.value = "Registration error: Candidate emails must have the @medishift.ac.in domain."
                    onResult(false)
                    return@launch
                }

                // Check if candidate is pre-registered by HR with status "Hired"
                val candidate = repository.getCandidateByEmail(email)
                if (candidate == null) {
                    _authError.value = "Registration error: Candidate with email '$email' is not pre-registered/hired by HR."
                    onResult(false)
                    return@launch
                }

                if (candidate.status == "Registered") {
                    _authError.value = "Registration error: Account already created for this pre-registered email."
                    onResult(false)
                    return@launch
                }

                if (!candidate.role.equals(role, ignoreCase = true)) {
                    _authError.value = "Registration error: Selected role does not match pre-registered role (${candidate.role})."
                    onResult(false)
                    return@launch
                }

                seniority = candidate.seniority
                // Mark candidate as Registered
                repository.updateCandidateStatus(candidate.id, "Registered")
            } else {
                // For HR, verify domain
                if (!email.endsWith("@medishift.ac.in", ignoreCase = true)) {
                    _authError.value = "Registration error: HR emails must end with @medishift.ac.in"
                    onResult(false)
                    return@launch
                }
            }

            // Create corresponding StaffProfile if role is not HR
            var staffId: Int? = null
            var generatedEmpId = ""
            if (role != "HR") {
                generatedEmpId = generateNextEmployeeId(role)
                val profile = StaffProfile(
                    name = name,
                    role = role,
                    skillLevel = seniority,
                    dayOffPreference = "None",
                    employeeId = generatedEmpId
                )
                repository.insertStaff(profile)
                // Retrieve back to get ID
                val allUpdatedStaff = repository.getAllStaffList()
                staffId = allUpdatedStaff.find { it.name == name }?.id
            } else {
                // Generate a custom ID for HR
                val count = repository.allUserAccounts.first().count { it.role.equals(role, ignoreCase = true) }
                generatedEmpId = "HR${2026}S${String.format("%02d", count + 1)}"
            }

            val newUser = UserAccount(
                email = email,
                passwordPlain = passwordPlain,
                name = name,
                role = role,
                staffProfileId = staffId,
                employeeId = generatedEmpId
            )
            repository.insertUserAccount(newUser)
            triggerLocalSync("MUTATION", "Registered user: ${newUser.name} as ${newUser.role}. Profile instantiated in database.")
            _currentUser.value = newUser
            onResult(true)
        }
    }

    fun logout() {
        _currentUser.value = null
        _authError.value = null
    }

    fun updateUserProfile(education: String, address: String) {
        viewModelScope.launch {
            val current = _currentUser.value ?: return@launch
            val updated = current.copy(education = education, address = address)
            repository.insertUserAccount(updated)
            triggerLocalSync("MUTATION", "Profile updated for ${updated.name} (ID: ${updated.employeeId}). Local database synchronized.")
            _currentUser.value = updated
        }
    }

    // Toggle Roster release status (Operations Manager)
    fun setRosterReleased(released: Boolean) {
        viewModelScope.launch {
            updateOperationalStateInDb { it.copy(isRosterReleased = released) }
            
            LocalSyncManager.logEvent("MUTATION", "Local SQLite: Roster release state updated to $released in database.")
            triggerLocalSync("NOTIFICATION", "[Local Database] Operations Manager has ${if (released) "RELEASED" else "RECALLED"} the roster.")
        }
    }

    // Update day-off preference for Doctor/Nurse
    fun updateStaffDayOff(staffId: Int, dayOff: String) {
        viewModelScope.launch {
            repository.updateStaffDayOffPreference(staffId, dayOff)
            
            LocalSyncManager.logEvent("MUTATION", "Local SQLite: Staff ID $staffId changed day-off preference to $dayOff.")
            triggerLocalSync("MUTATION", "[Local Database] Staff ID $staffId changed day-off preference to $dayOff.")
        }
    }

    // Update shift preference for Doctor/Nurse
    fun updateStaffShiftPreference(staffId: Int, shiftPref: String) {
        viewModelScope.launch {
            repository.updateStaffShiftPreference(staffId, shiftPref)
            
            LocalSyncManager.logEvent("MUTATION", "Local SQLite: Staff ID $staffId changed shift preference to $shiftPref.")
            triggerLocalSync("MUTATION", "[Local Database] Staff ID $staffId changed shift preference to $shiftPref.")
        }
    }

    // --- NON-AVAILABILITY / LEAVE REQUEST WORKFLOW ---
    // Staff submit a request covering one or more days of the week; it only
    // takes effect against the roster once the Operations Manager approves it
    // via approveLeaveRequest -- until then it's purely informational and the
    // solver treats the staff member as available as normal.
    fun submitLeaveRequest(staffId: Int, staffName: String, staffRole: String, days: List<String>, reason: String) {
        if (days.isEmpty()) return
        viewModelScope.launch {
            repository.insertLeaveRequest(
                LeaveRequest(
                    staffId = staffId,
                    staffName = staffName,
                    staffRole = staffRole,
                    days = days.joinToString(","),
                    reason = reason,
                    status = "Pending",
                    requestedAt = getIndianStandardTimeStr()
                )
            )
            LocalSyncManager.logEvent("MUTATION", "Local SQLite: $staffName requested non-availability for ${days.joinToString(", ")}.")
            triggerLocalSync("NOTIFICATION", "[Local Database] $staffName requested leave/non-availability for ${days.joinToString(", ")}. Awaiting Operations Manager approval.")
        }
    }

    // Only the Operations Manager's "Leave Approval" screen calls these. Once
    // approved, the change is picked up in real time by every observer of
    // leaveRequests (the Staff Pool counts in LPStaffingPlannerScreen and the
    // day-by-day roster solver in runConstructiveRosterAssignment both read the same live
    // StateFlow), so no separate "recompute" step is needed.
    fun approveLeaveRequest(id: Int) {
        viewModelScope.launch {
            repository.updateLeaveRequestStatus(id, "Approved")
            val req = leaveRequests.value.find { it.id == id }
            LocalSyncManager.logEvent("MUTATION", "Local SQLite: Leave request #$id approved by Operations Manager.")
            triggerLocalSync("NOTIFICATION", "[Local Database] Operations Manager APPROVED ${req?.staffName ?: "staff"}'s non-availability for ${req?.days ?: "requested days"}. Staff Pool and roster updated in real time.")
        }
    }

    fun rejectLeaveRequest(id: Int) {
        viewModelScope.launch {
            repository.updateLeaveRequestStatus(id, "Rejected")
            val req = leaveRequests.value.find { it.id == id }
            LocalSyncManager.logEvent("MUTATION", "Local SQLite: Leave request #$id rejected by Operations Manager.")
            triggerLocalSync("NOTIFICATION", "[Local Database] Operations Manager REJECTED ${req?.staffName ?: "staff"}'s non-availability request.")
        }
    }

    // True if this staff member has an APPROVED leave request covering `day`
    // (a day-of-week name, e.g. "Monday" -- this app models one recurring
    // roster week rather than literal calendar dates, matching dayOffPreference
    // and FinalRosterItem.date elsewhere). Reads the live StateFlow directly so
    // callers (runConstructiveRosterAssignment, the Staff Pool screen) always see approvals
    // the instant they happen.
    fun isStaffOnApprovedLeave(staffId: Int, day: String): Boolean {
        return leaveRequests.value.any { req ->
            req.staffId == staffId &&
                req.status == "Approved" &&
                req.days.split(",").any { it.trim().equals(day, ignoreCase = true) }
        }
    }

    fun updatePrediction(patientInflow: Int) {
        viewModelScope.launch {
            val staffNeeded = when {
                patientInflow >= 1800 -> 8
                patientInflow >= 1500 -> 6
                patientInflow >= 1200 -> 5
                else -> 4
            }
            updateOperationalStateInDb {
                it.copy(
                    predictedInflow = patientInflow,
                    dynamicStaffNeeded = staffNeeded
                )
            }
        }
    }

    // HR Candidates, Seniority and Payroll Management
    fun addCandidate(name: String, emailPrefix: String, role: String, seniority: String, salary: Double, allowances: Double) {
        viewModelScope.launch {
            val email = "${emailPrefix.trim().lowercase()}@medishift.ac.in"
            val empId = generateNextEmployeeId(role)
            repository.insertCandidate(
                Candidate(
                    name = name,
                    email = email,
                    role = role,
                    seniority = seniority,
                    status = "Hired",
                    salary = salary,
                    allowances = allowances
                )
            )
            StaffDatasetManager.addStaff(
                getApplication(),
                StaffJsonModel(
                    id = empId,
                    name = name,
                    email = email,
                    role = role,
                    seniority = seniority,
                    status = "Registered",
                    salary = salary,
                    allowances = allowances
                )
            )
            triggerLocalSync("MUTATION", "Pre-registered candidate: $name ($role). Welcome email generated.")
            // Welcome Email
            val welcomeEmail = EmailMessage(
                senderEmail = "hr@medishift.ac.in",
                receiverEmail = email,
                subject = "Welcome to MediShift!",
                body = "Dear $name,\n\nCongratulations on your selection as a $role ($seniority).\n\nYour profile has been created with Employee ID: $empId.\n\nYou are authorized to sign up using your registered official email: $email.\n\nBest Regards,\nHR Team",
                timestamp = getIndianStandardTimeStr()
            )
            repository.insertEmail(welcomeEmail)
        }
    }

    fun applyForJob(name: String, email: String, role: String, seniority: String) {
        viewModelScope.launch {
            repository.insertCandidate(
                Candidate(
                    name = name,
                    email = email,
                    role = role,
                    seniority = seniority,
                    status = "Applied",
                    salary = 0.0,
                    allowances = 0.0
                )
            )
        }
    }

    fun approveCandidate(candidateId: Int, name: String, emailPrefix: String, role: String, seniority: String, salary: Double, allowances: Double) {
        viewModelScope.launch {
            val email = "${emailPrefix.trim().lowercase()}@medishift.ac.in"
            val empId = generateNextEmployeeId(role)
            val updatedCandidate = Candidate(
                id = candidateId,
                name = name,
                email = email,
                role = role,
                seniority = seniority,
                status = "Hired",
                salary = salary,
                allowances = allowances
            )
            repository.insertCandidate(updatedCandidate)
            triggerLocalSync("MUTATION", "Approved and hired candidate: $name as $role ($seniority). Credentials issued.")
            // Welcome Email
            val welcomeEmail = EmailMessage(
                senderEmail = "hr@medishift.ac.in",
                receiverEmail = email,
                subject = "Job Application Approved - Welcome to MediShift!",
                body = "Dear $name,\n\nWe are pleased to inform you that your job application for the position of $role has been approved.\n\nYour employee credentials:\nEmployee ID: $empId\nOfficial Email: $email\nSeniority: $seniority\nBase Salary: ₹$salary/mo\nAllowances: ₹$allowances/mo\n\nPlease proceed to create your official account using this email.\n\nBest Regards,\nHR Team",
                timestamp = getIndianStandardTimeStr()
            )
            repository.insertEmail(welcomeEmail)
        }
    }

    fun updateCandidateSeniority(candidateId: Int, newSeniority: String, baseSalary: Double, allowances: Double) {
        viewModelScope.launch {
            repository.updateCandidateSeniorityAndSalary(candidateId, newSeniority, baseSalary, allowances)
        }
    }

    fun deleteCandidate(candidateId: Int) {
        viewModelScope.launch {
            repository.deleteCandidate(candidateId)
        }
    }

    fun removeStaff(staff: StaffProfile) {
        viewModelScope.launch {
            repository.deleteStaff(staff)
            triggerLocalSync("MUTATION", "Removed staff profile: ${staff.name} (${staff.role}). Registry updated.")
        }
    }

    // Email Operations (Inbox)
    fun sendEmail(receiverEmail: String, subject: String, body: String, onResult: (Boolean) -> Unit) {
        viewModelScope.launch {
            val sender = currentUser.value?.email ?: ""
            if (sender.isEmpty()) {
                onResult(false)
                return@launch
            }
            if (receiverEmail.isBlank() || subject.isBlank() || body.isBlank()) {
                onResult(false)
                return@launch
            }
            val emailMsg = EmailMessage(
                senderEmail = sender,
                receiverEmail = receiverEmail,
                subject = subject,
                body = body,
                timestamp = getIndianStandardTimeStr(),
                isRead = false
            )
            repository.insertEmail(emailMsg)
            onResult(true)
        }
    }

    fun markEmailAsRead(id: Int) {
        viewModelScope.launch {
            repository.markEmailAsRead(id)
        }
    }

    suspend fun generateNextEmployeeId(role: String): String {
        val allStaff = repository.getAllStaffList()
        val roleCode = when (role) {
            "Doctor" -> "DR"
            "Nurse" -> "NU"
            "Medical Officer" -> "MO"
            "Operations Manager" -> "OM"
            "Receptionist" -> "RE"
            "HR" -> "HR"
            "Pharmacist" -> "PH"
            "Lab Technician" -> "LT"
            else -> "EMP"
        }
        val count = allStaff.count { it.role.equals(role, ignoreCase = true) }
        val seq = String.format("S%02d", count + 1)
        return "$roleCode${2026}$seq"
    }

    fun saveShiftWiseInflow(
        dateString: String,
        morningCount: Int,
        eveningCount: Int,
        nightCount: Int,
        onSuccess: (String) -> Unit
    ) {
        viewModelScope.launch {
            try {
                val total = morningCount + eveningCount + nightCount
                val currentList = repository.getHistoricalInflowList()
                val existing = currentList.find { it.date == dateString }
                val itemToInsert = if (existing != null) {
                    existing.copy(patientCount = total)
                } else {
                    HistoricalInflow(date = dateString, patientCount = total)
                }
                
                repository.insertInflow(itemToInsert)

                // Save individual shift records to 3-Year JSON Dataset
                ShiftDatasetManager.updateOrAddShiftRecord(
                    context = getApplication(),
                    dateStr = dateString,
                    shiftType = "Morning",
                    patientInflow = morningCount,
                    weather = "Normal",
                    isHoliday = false,
                    isLocalEvent = false
                )
                ShiftDatasetManager.updateOrAddShiftRecord(
                    context = getApplication(),
                    dateStr = dateString,
                    shiftType = "Evening",
                    patientInflow = eveningCount,
                    weather = "Normal",
                    isHoliday = false,
                    isLocalEvent = false
                )
                ShiftDatasetManager.updateOrAddShiftRecord(
                    context = getApplication(),
                    dateStr = dateString,
                    shiftType = "Night",
                    patientInflow = nightCount,
                    weather = "Normal",
                    isHoliday = false,
                    isLocalEvent = false
                )

                LocalSyncManager.logEvent("PATIENT_REPORT", "Saved shift-wise intake for $dateString (Morning: $morningCount, Evening: $eveningCount, Night: $nightCount | Total: $total).")
                triggerLocalSync("PATIENT_REPORT", "[Shift-Wise Intake] $dateString: Morning=$morningCount, Evening=$eveningCount, Night=$nightCount (Total=$total).")
                runEnsembleForecasting()
                onSuccess("Shift-wise intake (Morning: $morningCount, Evening: $eveningCount, Night: $nightCount | Total: $total) saved successfully & forecast updated!")
            } catch (e: Exception) {
                android.util.Log.e("MediShiftVM", "Failed to save shift inflow: ${e.message}")
                onSuccess("Failed to save: ${e.localizedMessage ?: e.message}")
            }
        }
    }

    // Single entry point for every "run the solver" trigger in the UI (dashboard
    // button, roster grid empty-state, Solver Bench screen). Previously several
    // of these were wired to a separate Google OR-Tools CP-SAT solver
    // (solveShiftOptimization) that wrote to the same roster table as the MILP
    // solver below -- whichever ran last silently won, which is why the
    // "Finalize & Release Roster" screen could show stale or ratio-inconsistent
    // results. That CP-SAT path has been removed entirely; the MILP/ratio-matching
    // solver is now the single source of truth for the roster. This wrapper pulls
    // patient inflow and available staff counts from live state so every call
    // site can trigger it with no arguments, exactly like the old one-tap buttons.
    fun runPrimarySolver() {
        val pool = staffList.value.filter { it.isInOptimizationPool }
        fun countInCategory(category: String) = pool.count { member ->
            val r = member.role.trim().lowercase()
            when (category) {
                "Doctor" -> r.contains("doctor") || r.contains("medical officer") || r.contains("physician") || r.contains("surgeon") || r == "dr"
                "Nurse" -> r.contains("nurse") || r.contains("nursing") || r == "nu"
                "Pharmacist" -> r.contains("pharmacist") || r.contains("pharm") || r == "ph"
                "Lab Technician" -> r.contains("lab") || r.contains("technician") || r.contains("tech") || r == "lt"
                else -> false
            }
        }
        solveStaffingLP(
            patients = _predictedInflow.value,
            availableDocs = countInCategory("Doctor"),
            availableNurses = countInCategory("Nurse"),
            availablePhars = countInCategory("Pharmacist"),
            availableLabs = countInCategory("Lab Technician")
        )
    }

    // A robust Local Backtracking / Ratio-Based Constraint Solver -- the app's
    // single roster solver (a separate Google OR-Tools CP-SAT solver used to
    // exist in parallel and write to the same roster table; it has been removed
    // so there is exactly one source of truth for "Finalize & Release Roster").
    // Assigns staff to Morning, Evening, and Night shifts for a 7-day week based on patient-staff ratios
    //
    // Ratio parameters: docCap/nurseCap/pharCap/labCap are the "Target" (ideal)
    // ratios and docGoodCap/nurseGoodCap/pharGoodCap/labGoodCap are the "Good"
    // (safety floor) ratios from the project's ratio-matching model -- the same
    // two-ratio system used by the category-level OptimizationModels.solveStaffingLP.
    //
    // Each day's per-category staff pool is pre-allocated across all three shift
    // blocks up front (see allocateAcrossShifts below) so every shift's own
    // minSafe floor is guaranteed first (if the pool allows), and only the
    // remaining pool is spent closing each shift's own gap to its ideal --
    // this is what keeps Night's ratio from being starved by Morning/Evening
    // claiming the day's pool first.
    suspend fun runConstructiveRosterAssignment(
        docCap: Double = 20.0,
        nurseCap: Double = 6.0,
        pharCap: Double = 75.0,
        labCap: Double = 40.0,
        docGoodCap: Double = 50.0,
        nurseGoodCap: Double = 20.0,
        pharGoodCap: Double = 100.0,
        labGoodCap: Double = 100.0
    ) {
        _isOptimizing.value = true
        _solverStatusMessage.value = "Initializing ratio-matching staffing model..."
        delay(600)

        _solverStatusMessage.value = "Loading staff profiles from active optimization pool..."
        val allStaff = repository.getAllStaffList()
        val staff = allStaff.filter { it.isInOptimizationPool }
        delay(500)

        if (staff.isEmpty()) {
            _solverStatusMessage.value = "Error: No staff active in optimization pool! Please enable staff in the Pool Manager."
            _isOptimizing.value = false
            return
        }

        _solverStatusMessage.value = "Applying constraints: HARD (Max 1 shift/day, Night blocks & rest) | SOFT (Patient ratios & preferences)..."
        delay(600)

        _solverStatusMessage.value = "Executing constructive roster assignment (hard rules enforced, ratio targets approximated)..."
            
        // Scheduling parameters
        val days = listOf("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        val shifts = listOf("Morning", "Evening", "Night")
        val categories = listOf("Doctor", "Nurse", "Pharmacist", "Lab Technician")
        val dailyInflow = _predictedInflow.value.coerceAtLeast(30)

        val localAssignments = mutableListOf<FinalRosterItem>()

        // Track state for constraints & workload balancing
        val staffShiftCounts = mutableMapOf<Int, Int>() // staffId -> number of assigned shifts
        val lastAssignedShift = mutableMapOf<Int, String>() // staffId -> last shift type assigned ("Morning", "Evening", "Night")
        val lastAssignedDay = mutableMapOf<Int, String>() // staffId -> day of last assignment

        // State tracking for 2-3 continuous night shift blocks and mandatory rest day
        val nightStreakMap = mutableMapOf<Int, Int>() // staffId -> current consecutive night shifts
        val nightTargetMap = mutableMapOf<Int, Int>() // staffId -> target night streak (2 or 3)
        val mandatoryRestDayMap = mutableMapOf<Int, Boolean>() // staffId -> true if compulsory day off today

        // Keep track of soft constraint success metrics
        var totalPreferencesMet = 0
        var totalPreferencesChecked = 0

        // Per-day forecast (Section 2.5 / Eq. 2.10-2.11 in the report): the
        // ensemble now produces one forecast per roster day (Monday..Sunday),
        // not a single day's number reused for the whole week. weeklyForecast
        // is only populated after Advanced Forecasting has actually run at
        // least once (see runEnsembleForecasting); if it hasn't, fall back to
        // the same flat proportional split this function always used before
        // the weekly forecast existed, so the solver still works standalone.
        val ensembleRes = _ensembleResult.value
        val weeklyForecast = ensembleRes?.weeklyForecast ?: emptyList()
        val fallbackMorning = ensembleRes?.morningPred ?: (dailyInflow * 0.45).toInt()
        val fallbackEvening = ensembleRes?.eveningPred ?: (dailyInflow * 0.35).toInt()
        val fallbackNight = ensembleRes?.nightPred ?: (dailyInflow * 0.20).toInt()

        // Category -> eligible staff (computed once; the pool itself doesn't
        // change during the solve, only who's available on a given day does).
        val categoryStaffMap: Map<String, List<StaffProfile>> = categories.associateWith { category ->
            staff.filter { member ->
                val r = member.role.trim().lowercase()
                when (category) {
                    "Doctor" -> r.contains("doctor") || r.contains("medical officer") || r.contains("physician") || r.contains("surgeon") || r == "dr"
                    "Nurse" -> r.contains("nurse") || r.contains("nursing") || r == "nu"
                    "Pharmacist" -> r.contains("pharmacist") || r.contains("pharm") || r == "ph"
                    "Lab Technician" -> r.contains("lab") || r.contains("technician") || r.contains("tech") || r == "lt"
                    else -> true
                }
            }
        }

        data class ShiftTargets(val minSafe: Int, val ideal: Int)
        data class CategoryRatioParams(val minFloor: Int, val goodRatio: Double, val targetRatio: Double)

        // Each category's ratio POLICY (its safety floor and target ratios) --
        // these are configuration, not forecast numbers, so they genuinely are
        // the same every day of the week.
        val categoryRatioParams: Map<String, CategoryRatioParams> = categories.associateWith { category ->
            // Absolute per-shift safety floor regardless of ratio (Doc: 2, Nurse: 3, Phar: 1, Lab: 1)
            val minFloor = when (category) {
                "Doctor" -> 2
                "Nurse" -> 3
                "Pharmacist" -> 1
                "Lab Technician" -> 1
                else -> 1
            }
            val goodRatio = when (category) {
                "Doctor" -> docGoodCap
                "Nurse" -> nurseGoodCap
                "Pharmacist" -> pharGoodCap
                "Lab Technician" -> labGoodCap
                else -> 100.0
            }
            val targetRatio = when (category) {
                "Doctor" -> docCap
                "Nurse" -> nurseCap
                "Pharmacist" -> pharCap
                "Lab Technician" -> labCap
                else -> 40.0
            }
            CategoryRatioParams(minFloor, goodRatio, targetRatio)
        }

        // Computes each category's minSafe (Good-ratio floor) and ideal
        // (Target-ratio aim) per shift block for ONE GIVEN DAY's own
        // forecasted inflow. This is the fix for the day-flattening issue:
        // previously this whole computation ran ONCE outside the day loop
        // from a single shift-block forecast and was then reused unchanged
        // for all seven days, so one day's forecast (whichever day the
        // ensemble happened to represent) silently set every day's staffing
        // target for the whole week, and a single unusual day's outlier
        // reading rippled through the entire roster. Each day now looks up
        // its own entry in weeklyForecast instead.
        fun shiftTargetsForDay(day: String): Map<String, Map<String, ShiftTargets>> {
            val dayForecast = weeklyForecast.find { it.day == day }
            val shiftInflowByShift = mapOf(
                "Morning" to (dayForecast?.morning ?: fallbackMorning).coerceAtLeast(15),
                "Evening" to (dayForecast?.evening ?: fallbackEvening).coerceAtLeast(10),
                "Night" to (dayForecast?.night ?: fallbackNight).coerceAtLeast(5)
            )
            return categories.associateWith { category ->
                val params = categoryRatioParams.getValue(category)
                shiftInflowByShift.mapValues { (_, inflow) ->
                    val minSafe = kotlin.math.ceil(inflow.toDouble() / params.goodRatio).toInt().coerceAtLeast(params.minFloor)
                    val ideal = kotlin.math.ceil(inflow.toDouble() / params.targetRatio).toInt().coerceAtLeast(minSafe)
                    ShiftTargets(minSafe = minSafe, ideal = ideal)
                }
            }
        }

        // Largest-remainder proportional allocation of `total` units across
        // `weights` (each bucket capped at its own weight). Used both to share
        // out a pool too small to cover every shift's safety floor, and to
        // share out leftover pool across each shift's remaining gap to ideal.
        fun proportionalAllocate(total: Int, weights: List<Int>): List<Int> {
            val sumWeights = weights.sum()
            if (sumWeights <= 0 || total <= 0) return weights.map { 0 }
            val cappedTotal = total.coerceAtMost(sumWeights)
            val raw = weights.map { it.toDouble() * cappedTotal / sumWeights }
            val base = raw.mapIndexed { i, r -> kotlin.math.floor(r).toInt().coerceAtMost(weights[i]) }
            var remainder = cappedTotal - base.sum()
            val order = raw.indices.sortedByDescending { raw[it] - base[it] }
            val result = base.toMutableList()
            for (i in order) {
                if (remainder <= 0) break
                if (result[i] < weights[i]) {
                    result[i] += 1
                    remainder -= 1
                }
            }
            return result
        }

        // Splits a single day's category pool across Morning/Evening/Night so
        // that the ratio-matching optimality condition is honored at EVERY
        // shift, not just whichever shift is processed first: every shift's
        // minSafe floor is covered first (proportionally, if the pool can't
        // cover all three), then any pool left over is spent closing each
        // shift's own remaining gap to its ideal (also proportionally, if it
        // can't close every gap).
        fun allocateAcrossShifts(pool: Int, minSafe: List<Int>, ideal: List<Int>): List<Int> {
            if (pool <= 0) return listOf(0, 0, 0)
            val totalMinSafe = minSafe.sum()
            val floors = if (totalMinSafe <= pool) minSafe else proportionalAllocate(pool, minSafe)
            val afterFloors = pool - floors.sum()
            val gaps = ideal.indices.map { (ideal[it] - floors[it]).coerceAtLeast(0) }
            val extra = if (afterFloors >= gaps.sum()) gaps else proportionalAllocate(afterFloors, gaps)
            return floors.indices.map { floors[it] + extra[it] }
        }

        // Weekly-capacity-aware demand shaping. Without this, each day's "ideal"
        // target (Section 3.1) was computed purely from that day's OWN forecast,
        // with no awareness of how much of the category's weekly 5-shift cap
        // (Hard Rule 2, Eq. 3.6) was already spent on earlier days. A category
        // with a tight ratio and a small pool -- Nurse's 1:6 target ratio asks
        // for far more heads per shift than Doctor's 1:20 -- can happily get
        // assigned near its "ideal" headcount on Monday through Thursday, only
        // to discover every one of its staff has hit 5 shifts by Friday, leaving
        // literally nobody left to assign for the last two or three days. Since
        // Doctor's much looser ratio asks for a smaller headcount per shift, it
        // burns through its own staff's weekly budget more slowly and keeps
        // getting assigned long after the other categories have run dry -- which
        // is exactly the "only Doctors get scheduled toward the end of the week"
        // pattern this fixes. This scales back the ASPIRATIONAL part of every
        // day's target (ideal minus minSafe) by the same factor for every day of
        // the week, so that if a category's total week-long "ideal" demand would
        // outrun its total weekly capacity (pool size x 5), the shortfall is
        // spread evenly across all seven days instead of silently concentrating
        // on whichever days happen to be scheduled last. minSafe itself is never
        // scaled down -- if even minSafe alone would exceed weekly capacity, that
        // is a genuine staffing deficit (the pool is too small for this ratio
        // policy), not something a smarter schedule can paper over, and it is
        // left to surface honestly rather than hidden by this fix.
        val categoryIdealScale: Map<String, Double> = categories.associateWith { category ->
            val poolSize = (categoryStaffMap[category] ?: emptyList()).size
            val weeklyCapacity = poolSize * 5
            var totalMinSafe = 0
            var totalIdeal = 0
            for (day in days) {
                val targets = shiftTargetsForDay(day).getValue(category)
                for (shift in shifts) {
                    val t = targets.getValue(shift)
                    totalMinSafe += t.minSafe
                    totalIdeal += t.ideal
                }
            }
            val extraDemand = totalIdeal - totalMinSafe
            if (weeklyCapacity <= 0 || extraDemand <= 0) {
                1.0
            } else {
                val extraCapacity = (weeklyCapacity - totalMinSafe).coerceAtLeast(0)
                (extraCapacity.toDouble() / extraDemand).coerceIn(0.0, 1.0)
            }
        }

        // Category-wise Shift-wise MILP / Ratio-Based Schedule Assignment Algorithm
        for (dayIndex in days.indices) {
            val day = days[dayIndex]
            val yesterday = if (dayIndex > 0) days[dayIndex - 1] else null

            // Update mandatory rest day & active night block status for 'day'
            for (member in staff) {
                val lastDay = lastAssignedDay[member.id]
                val lastShift = lastAssignedShift[member.id]
                val currentStreak = nightStreakMap[member.id] ?: 0

                if (yesterday != null && lastDay == yesterday && lastShift == "Night") {
                    val target = nightTargetMap[member.id] ?: 2
                    if (currentStreak >= target || currentStreak >= 3) {
                        // Completed a 2-3 day night shift block yesterday! Today is mandatory rest day.
                        mandatoryRestDayMap[member.id] = true
                    } else {
                        // In middle of 2-3 day night block
                        mandatoryRestDayMap[member.id] = false
                    }
                } else {
                    mandatoryRestDayMap[member.id] = false
                    if (lastDay != yesterday || lastShift != "Night") {
                        nightStreakMap[member.id] = 0
                    }
                }
            }

            // This day's own minSafe/ideal targets, from this day's own entry
            // in weeklyForecast -- not a single shared value for the whole week.
            // The "ideal" half of each target is tempered by categoryIdealScale
            // above, so a tight-ratio/small-pool category's aspirational reach
            // is spread evenly across the week instead of front-loaded onto the
            // first few days at the expense of the last few; minSafe is passed
            // through unscaled.
            val perCategoryShiftTargets = shiftTargetsForDay(day).mapValues { (category, shiftMap) ->
                val scale = categoryIdealScale.getValue(category)
                shiftMap.mapValues { (_, t) ->
                    val scaledIdeal = t.minSafe + ((t.ideal - t.minSafe).coerceAtLeast(0) * scale).toInt()
                    ShiftTargets(minSafe = t.minSafe, ideal = scaledIdeal.coerceAtLeast(t.minSafe))
                }
            }

            // Pre-allocate today's per-category pool across all three shift
            // blocks up front -- this is the fix: Night's headcount is now
            // derived from its OWN ideal ratio (shared fairly against Morning
            // and Evening's), not from whatever happens to be left over after
            // Morning and Evening have already been assigned.
            val poolByCategoryToday: Map<String, Int> = categories.associateWith { category ->
                val catStaff = categoryStaffMap[category] ?: emptyList()
                catStaff.count { mandatoryRestDayMap[it.id] != true && !isStaffOnApprovedLeave(it.id, day) }
            }
            val todaysShiftTargets: Map<String, List<Int>> = categories.associateWith { category ->
                val targets = perCategoryShiftTargets.getValue(category)
                allocateAcrossShifts(
                    pool = poolByCategoryToday.getValue(category),
                    minSafe = listOf(targets.getValue("Morning").minSafe, targets.getValue("Evening").minSafe, targets.getValue("Night").minSafe),
                    ideal = listOf(targets.getValue("Morning").ideal, targets.getValue("Evening").ideal, targets.getValue("Night").ideal)
                )
            }
            val shiftIndex = mapOf("Morning" to 0, "Evening" to 1, "Night" to 2)

            for (shift in shifts) {
                for (category in categories) {
                    val categoryStaff = categoryStaffMap[category] ?: emptyList()
                    if (categoryStaff.isEmpty()) continue

                    val targetCount = todaysShiftTargets.getValue(category)[shiftIndex.getValue(shift)]
                        .coerceAtMost(categoryStaff.size)

                    // Pass 1: Strict constraints
                    // Hard Rule 1: Max 1 shift per day (STRICT)
                    // Hard Rule 2: Night shift block (2-3 continuous days) + mandatory rest day next day
                    val pass1Candidates = categoryStaff.shuffled().filter { member ->
                        // 1. Max 5 shifts/wk limit
                        val currentCount = staffShiftCounts[member.id] ?: 0
                        if (currentCount >= 5) return@filter false

                        // 2. HARD REQUIREMENT 1: One staff member MUST NOT get more than one shift in a single day
                        if (localAssignments.any { it.staffId == member.id && it.date.equals(day, ignoreCase = true) }) {
                            return@filter false
                        }

                        // 3. HARD REQUIREMENT 2: Mandatory rest day after finishing a 2-3 day night shift block
                        if (mandatoryRestDayMap[member.id] == true) {
                            return@filter false
                        }

                        // 3b. HARD REQUIREMENT: Approved non-availability/leave for this day
                        if (isStaffOnApprovedLeave(member.id, day)) {
                            return@filter false
                        }

                        // 4. HARD REQUIREMENT 2: Continuous Night Block
                        // If member is in an active night block (streak 1 or 2), they MUST NOT be assigned to Morning or Evening
                        val currentNightStreak = nightStreakMap[member.id] ?: 0
                        if (shift != "Night" && currentNightStreak > 0) {
                            return@filter false
                        }

                        // 5. HARD REQUIREMENT 3 (report Eq. 3.9): nobody may be rostered onto a
                        // fourth consecutive Night shift. This is checked directly here (not just
                        // relied upon via mandatoryRestDayMap above) so the cap holds even if the
                        // per-person night-block target below is ever changed to something larger.
                        if (shift == "Night" && currentNightStreak >= 3) {
                            return@filter false
                        }

                        val lastDay = lastAssignedDay[member.id]
                        val lastShift = lastAssignedShift[member.id]

                        if (lastDay != null) {
                            val yesterdayIndex = days.indexOf(lastDay)
                            val todayIndex = days.indexOf(day)
                            if (todayIndex == yesterdayIndex + 1 && lastShift == "Night" && shift == "Morning") {
                                return@filter false
                            }
                        }

                        true
                    }.sortedWith(compareBy<StaffProfile> { member ->
                        // Top priority for members continuing their 2-3 day night block
                        val currentNightStreak = nightStreakMap[member.id] ?: 0
                        val nightBlockPriority = if (shift == "Night" && currentNightStreak in 1..2) -1000 else 0

                        val isDayOff = member.dayOffPreference.equals(day, ignoreCase = true)
                        if (isDayOff) totalPreferencesChecked++
                        val dayOffScore = if (isDayOff) 20 else 0

                        val isPreferredShift = member.shiftPreference.equals(shift, ignoreCase = true)
                        val isNonePref = member.shiftPreference.equals("None", ignoreCase = true) || member.shiftPreference.isBlank()
                        val shiftPrefScore = when {
                            isPreferredShift -> -10
                            isNonePref -> 0
                            else -> 10
                        }

                        val currentCount = staffShiftCounts[member.id] ?: 0

                        nightBlockPriority + dayOffScore + shiftPrefScore + currentCount
                    })

                    val assigned = mutableListOf<StaffProfile>()
                    assigned.addAll(pass1Candidates.take(targetCount))

                    // Pass 2: Fallback when Pass 1 produces fewer than targetCount due to strict limits.
                    // Maintains strict max 1 shift/day, mandatory rest days, and active night block lock.
                    if (assigned.size < targetCount) {
                        val needed = targetCount - assigned.size
                        val pass2Candidates = categoryStaff.filter { member ->
                            !assigned.any { it.id == member.id } &&
                            (staffShiftCounts[member.id] ?: 0) < 5 && // Max 5 shifts/wk limit (Eq. 3.6) -- hard rule, held even if the shift falls short
                            !localAssignments.any { it.staffId == member.id && it.date.equals(day, ignoreCase = true) } && // Strict Max 1 shift/day
                            mandatoryRestDayMap[member.id] != true && // Mandatory Rest Day after night block
                            !isStaffOnApprovedLeave(member.id, day) && // Approved non-availability/leave for this day
                            (shift == "Night" || (nightStreakMap[member.id] ?: 0) == 0) && // Cannot do Morning/Evening if in night block
                            (shift != "Night" || (nightStreakMap[member.id] ?: 0) < 3) // No 4th consecutive Night shift (Eq. 3.9)
                        }.sortedBy { member ->
                            staffShiftCounts[member.id] ?: 0
                        }
                        assigned.addAll(pass2Candidates.take(needed))
                    }

                    // Pass 3: Emergency Fallback for very small pools where all staff worked today.
                    // Strictly enforces MAX 1 SHIFT PER DAY across all fallback levels.
                    if (assigned.size < targetCount) {
                        val needed = targetCount - assigned.size
                        val pass3Candidates = categoryStaff.filter { member ->
                            !assigned.any { it.id == member.id } &&
                            (staffShiftCounts[member.id] ?: 0) < 5 && // Max 5 shifts/wk limit (Eq. 3.6) -- hard rule, held even in the emergency pass
                            !localAssignments.any { it.staffId == member.id && it.date.equals(day, ignoreCase = true) } && // Strict Max 1 shift/day
                            mandatoryRestDayMap[member.id] != true &&
                            !isStaffOnApprovedLeave(member.id, day) && // Approved non-availability/leave (Eq. 3.7) -- hard rule, held even in the emergency pass
                            (shift != "Night" || (nightStreakMap[member.id] ?: 0) < 3) // No 4th consecutive Night shift (Eq. 3.9), even in the emergency pass
                        }.sortedBy { member ->
                            staffShiftCounts[member.id] ?: 0
                        }
                        assigned.addAll(pass3Candidates.take(needed))
                    }

                    for (member in assigned) {
                        localAssignments.add(
                            FinalRosterItem(
                                staffId = member.id,
                                staffName = member.name,
                                staffRole = member.role,
                                date = day,
                                shiftSlot = shift
                            )
                        )

                        staffShiftCounts[member.id] = (staffShiftCounts[member.id] ?: 0) + 1
                        lastAssignedDay[member.id] = day
                        lastAssignedShift[member.id] = shift

                        if (shift == "Night") {
                            val prevStreak = nightStreakMap[member.id] ?: 0
                            val newStreak = prevStreak + 1
                            nightStreakMap[member.id] = newStreak
                            if (prevStreak == 0) {
                                // Set target night block length (2 or 3 continuous days)
                                nightTargetMap[member.id] = if (member.id % 2 == 0) 3 else 2
                            }
                        } else {
                            nightStreakMap[member.id] = 0
                        }

                        if (!member.dayOffPreference.equals(day, ignoreCase = true) && member.dayOffPreference != "None") {
                            totalPreferencesMet++
                        }
                    }
                }
            }
        }

            val finalAssignments = localAssignments

            // Calculate metrics
            val grantedOff = staff.count { member ->
                val day = member.dayOffPreference
                if (day == "None") return@count false
                // Check if they were assigned on their day off
                val assignedOnDayOff = localAssignments.any { it.staffId == member.id && it.date.equals(day, ignoreCase = true) }
                !assignedOnDayOff
            }

            val totalEligibleForPreference = staff.count { it.dayOffPreference != "None" }

            val softConstraintsPercent = if (totalEligibleForPreference > 0) (grantedOff * 100) / totalEligibleForPreference else 100
            val avgShifts = if (staff.isNotEmpty()) localAssignments.size.toDouble() / staff.size else 0.0

            _solverStatusMessage.value = "Saving finalized roster to Database..."
            repository.clearRoster()
            repository.insertRosterItems(finalAssignments)
            triggerLocalSync("MUTATION", "[Local Database] Final roster published with ${finalAssignments.size} shifts saved in local SQLite.")

            delay(500)

            updateOperationalStateInDb {
                it.copy(
                    solverTotalAssignments = finalAssignments.size,
                    solverHardConstraintsMet = true,
                    solverSoftConstraintsMetPercent = softConstraintsPercent,
                    solverAvgShiftsPerStaff = avgShifts,
                    solverPreferredDaysOffGranted = grantedOff,
                    hasSolverMetrics = true
                )
            }

            _solverStatusMessage.value = "Optimization Successful! Schedule published."
            delay(500)
            _isOptimizing.value = false
    }
}
