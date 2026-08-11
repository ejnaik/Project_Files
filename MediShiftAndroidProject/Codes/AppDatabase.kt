package com.example.data

import androidx.room.Dao
import androidx.room.Database
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.RoomDatabase
import kotlinx.coroutines.flow.Flow

@Dao
interface MediShiftDao {
    // Staff Profiles
    @Query("SELECT * FROM staff_profiles ORDER BY id DESC")
    fun getAllStaff(): Flow<List<StaffProfile>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertStaff(staff: StaffProfile)

    @Delete
    suspend fun deleteStaff(staff: StaffProfile)

    @Query("SELECT * FROM staff_profiles")
    suspend fun getAllStaffList(): List<StaffProfile>

    // Historical Inflow
    @Query("SELECT * FROM historical_inflow ORDER BY date DESC")
    fun getAllInflows(): Flow<List<HistoricalInflow>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertInflow(inflow: HistoricalInflow)

    @Query("SELECT * FROM historical_inflow")
    suspend fun getHistoricalInflowList(): List<HistoricalInflow>

    // Final Roster
    @Query("SELECT * FROM final_roster ORDER BY date ASC, shiftSlot ASC")
    fun getAllRosterItems(): Flow<List<FinalRosterItem>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertRosterItems(items: List<FinalRosterItem>)

    @Query("DELETE FROM final_roster")
    suspend fun clearRoster()

    // User Accounts
    @Query("SELECT * FROM user_accounts WHERE email = :email LIMIT 1")
    suspend fun getUserByEmail(email: String): UserAccount?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUserAccount(user: UserAccount): Long

    @Query("SELECT * FROM user_accounts")
    fun getAllUserAccounts(): Flow<List<UserAccount>>

    @Query("UPDATE staff_profiles SET dayOffPreference = :dayOff WHERE id = :staffId")
    suspend fun updateStaffDayOffPreference(staffId: Int, dayOff: String)

    @Query("UPDATE staff_profiles SET shiftPreference = :shiftPref WHERE id = :staffId")
    suspend fun updateStaffShiftPreference(staffId: Int, shiftPref: String)

    // Appointments
    @Query("SELECT * FROM appointments ORDER BY id DESC")
    fun getAllAppointments(): Flow<List<Appointment>>

    // Candidates (HR & Finance)
    @Query("SELECT * FROM candidates ORDER BY id DESC")
    fun getAllCandidates(): Flow<List<Candidate>>

    @Query("SELECT * FROM candidates")
    suspend fun getAllCandidatesList(): List<Candidate>

    @Query("SELECT * FROM candidates WHERE email = :email LIMIT 1")
    suspend fun getCandidateByEmail(email: String): Candidate?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCandidate(candidate: Candidate): Long

    @Query("UPDATE candidates SET status = :status WHERE id = :id")
    suspend fun updateCandidateStatus(id: Int, status: String)

    @Query("UPDATE candidates SET seniority = :seniority, salary = :salary, allowances = :allowances WHERE id = :id")
    suspend fun updateCandidateSeniorityAndSalary(id: Int, seniority: String, salary: Double, allowances: Double)

    @Query("DELETE FROM candidates WHERE id = :id")
    suspend fun deleteCandidate(id: Int)

    // Emails (Inbox)
    @Query("SELECT * FROM emails WHERE receiverEmail = :email ORDER BY id DESC")
    fun getEmailsForUser(email: String): Flow<List<EmailMessage>>

    @Query("SELECT * FROM emails WHERE receiverEmail = :email")
    suspend fun getEmailList(email: String): List<EmailMessage>

    @Query("SELECT * FROM emails WHERE senderEmail = :email ORDER BY id DESC")
    fun getSentEmailsForUser(email: String): Flow<List<EmailMessage>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEmail(email: EmailMessage): Long

    @Query("UPDATE emails SET isRead = 1 WHERE id = :id")
    suspend fun markEmailAsRead(id: Int)

    // Leave / Non-Availability Requests
    @Query("SELECT * FROM leave_requests ORDER BY id DESC")
    fun getAllLeaveRequests(): Flow<List<LeaveRequest>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertLeaveRequest(leaveRequest: LeaveRequest): Long

    @Query("UPDATE leave_requests SET status = :status WHERE id = :id")
    suspend fun updateLeaveRequestStatus(id: Int, status: String)

    // Centralized Operational State
    @Query("SELECT * FROM operational_state WHERE id = 1 LIMIT 1")
    fun getOperationalStateFlow(): Flow<OperationalState?>

    @Query("SELECT * FROM operational_state WHERE id = 1 LIMIT 1")
    suspend fun getOperationalState(): OperationalState?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertOperationalState(state: OperationalState)
}

@Database(
    entities = [
        StaffProfile::class,
        HistoricalInflow::class,
        FinalRosterItem::class,
        UserAccount::class,
        Appointment::class,
        Candidate::class,
        EmailMessage::class,
        OperationalState::class,
        LeaveRequest::class
    ],
    version = 11,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun dao(): MediShiftDao
}
