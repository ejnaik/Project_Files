package com.example.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "staff_profiles")
data class StaffProfile(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val name: String,
    val role: String, // "Doctor", "Nurse", "Pharmacist", "Lab Technician"
    val skillLevel: String, // "Senior", "Mid", "Junior"
    val dayOffPreference: String, // "None", "Monday", "Tuesday", etc.
    val employeeId: String = "",
    val shiftPreference: String = "None", // "None", "Morning", "Evening", "Night"
    val isInOptimizationPool: Boolean = true,
    val hourlyWage: Double = 500.0,
    val department: String = "General Clinical Services"
)

@Entity(tableName = "historical_inflow")
data class HistoricalInflow(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val date: String,
    val patientCount: Int
)

@Entity(tableName = "final_roster")
data class FinalRosterItem(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val staffId: Int,
    val staffName: String,
    val staffRole: String,
    val date: String, // "YYYY-MM-DD"
    val shiftSlot: String // "Morning", "Evening", "Night"
)

@Entity(tableName = "user_accounts")
data class UserAccount(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val email: String, // unique ID (such as an email ID)
    val passwordPlain: String, // plain password for simple client-side local DB matching
    val name: String,
    val role: String, // "Doctor", "Nurse", "Operations Manager", "Medical Officer", "Receptionist"
    val staffProfileId: Int? = null, // nullable staff connection
    val employeeId: String = "",
    val education: String = "",
    val address: String = ""
)

@Entity(tableName = "appointments")
data class Appointment(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val patientName: String,
    val appointmentTime: String, // e.g. "09:00 AM"
    val appointmentDate: String, // e.g. "Monday"
    val doctorName: String,
    val type: String, // "Online" or "Offline"
    val status: String // "Scheduled", "Completed", "Cancelled"
)

@Entity(tableName = "candidates")
data class Candidate(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val name: String,
    val email: String, // format: <something>@medishift.ac.in
    val role: String, // "Doctor", "Nurse", "Operations Manager", "Medical Officer", "Receptionist", "HR", "Pharmacist", "Lab Technician"
    val seniority: String, // "Junior", "Senior"
    val status: String, // "Hired" or "Registered"
    val salary: Double, // Monthly Base Salary
    val allowances: Double // Monthly Allowances
)

@Entity(tableName = "emails")
data class EmailMessage(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val senderEmail: String,
    val receiverEmail: String,
    val subject: String,
    val body: String,
    val timestamp: String, // format: "YYYY-MM-DD HH:mm"
    val isRead: Boolean = false
)

@Entity(tableName = "leave_requests")
data class LeaveRequest(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val staffId: Int,
    val staffName: String,
    val staffRole: String,
    val days: String, // comma-separated day names, e.g. "Monday,Wednesday,Friday"
    val reason: String = "",
    val status: String = "Pending", // "Pending", "Approved", "Rejected"
    val requestedAt: String = "" // "YYYY-MM-DD HH:mm"
)

@Entity(tableName = "operational_state")
data class OperationalState(
    @PrimaryKey val id: Int = 1,
    val predictedInflow: Int = 1580,
    val dynamicStaffNeeded: Int = 6,
    val isRosterReleased: Boolean = false,
    val solverTotalAssignments: Int = 0,
    val solverHardConstraintsMet: Boolean = false,
    val solverSoftConstraintsMetPercent: Int = 0,
    val solverAvgShiftsPerStaff: Double = 0.0,
    val solverPreferredDaysOffGranted: Int = 0,
    val hasSolverMetrics: Boolean = false,
    // Persisted snapshot of the ratio-matching staffing result (solveStaffingLP),
    // so the last computed headcounts/costs survive an app restart and can be
    // restored into lpResult without recomputation. (Renamed from the old
    // knapsack* field names left over from a now-removed, unused work-hour
    // knapsack solver; the underlying persistence role is unchanged.)
    val persistedBudget: Double = 500000.0,
    val persistedDoctors: Int = 0,
    val persistedNurses: Int = 0,
    val persistedPharmacists: Int = 0,
    val persistedLabTechs: Int = 0,
    val persistedTotalCost: Double = 0.0,
    val persistedTotalUtility: Double = 0.0,
    val hasPersistedResult: Boolean = false
)


