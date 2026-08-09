package com.example.data

import kotlinx.coroutines.flow.Flow

class MediShiftRepository(private val dao: MediShiftDao) {
    val allStaff: Flow<List<StaffProfile>> = dao.getAllStaff()
    val allInflows: Flow<List<HistoricalInflow>> = dao.getAllInflows()
    val allRosterItems: Flow<List<FinalRosterItem>> = dao.getAllRosterItems()

    suspend fun getAllStaffList(): List<StaffProfile> = dao.getAllStaffList()
    suspend fun getHistoricalInflowList(): List<HistoricalInflow> = dao.getHistoricalInflowList()

    suspend fun insertStaff(staff: StaffProfile) = dao.insertStaff(staff)
    suspend fun deleteStaff(staff: StaffProfile) = dao.deleteStaff(staff)

    suspend fun insertInflow(inflow: HistoricalInflow) = dao.insertInflow(inflow)

    suspend fun insertRosterItems(items: List<FinalRosterItem>) = dao.insertRosterItems(items)
    suspend fun clearRoster() = dao.clearRoster()

    // User Accounts
    val allUserAccounts: Flow<List<UserAccount>> = dao.getAllUserAccounts()
    suspend fun getUserByEmail(email: String): UserAccount? = dao.getUserByEmail(email)
    suspend fun insertUserAccount(user: UserAccount): Long = dao.insertUserAccount(user)
    suspend fun updateStaffDayOffPreference(staffId: Int, dayOff: String) = dao.updateStaffDayOffPreference(staffId, dayOff)
    suspend fun updateStaffShiftPreference(staffId: Int, shiftPref: String) = dao.updateStaffShiftPreference(staffId, shiftPref)

    // Appointments
    val allAppointments: Flow<List<Appointment>> = dao.getAllAppointments()

    // Candidates (HR & Finance)
    val allCandidates: Flow<List<Candidate>> = dao.getAllCandidates()
    suspend fun getAllCandidatesList(): List<Candidate> = dao.getAllCandidatesList()
    suspend fun getCandidateByEmail(email: String): Candidate? = dao.getCandidateByEmail(email)
    suspend fun insertCandidate(candidate: Candidate): Long = dao.insertCandidate(candidate)
    suspend fun updateCandidateStatus(id: Int, status: String) = dao.updateCandidateStatus(id, status)
    suspend fun updateCandidateSeniorityAndSalary(id: Int, seniority: String, salary: Double, allowances: Double) = dao.updateCandidateSeniorityAndSalary(id, seniority, salary, allowances)
    suspend fun deleteCandidate(id: Int) = dao.deleteCandidate(id)

    // Emails (Inbox)
    fun getEmailsForUser(email: String): Flow<List<EmailMessage>> = dao.getEmailsForUser(email)
    suspend fun getEmailList(email: String): List<EmailMessage> = dao.getEmailList(email)
    fun getSentEmailsForUser(email: String): Flow<List<EmailMessage>> = dao.getSentEmailsForUser(email)
    suspend fun insertEmail(email: EmailMessage): Long = dao.insertEmail(email)
    suspend fun markEmailAsRead(id: Int) = dao.markEmailAsRead(id)

    // Leave / Non-Availability Requests
    val allLeaveRequests: Flow<List<LeaveRequest>> = dao.getAllLeaveRequests()
    suspend fun insertLeaveRequest(leaveRequest: LeaveRequest): Long = dao.insertLeaveRequest(leaveRequest)
    suspend fun updateLeaveRequestStatus(id: Int, status: String) = dao.updateLeaveRequestStatus(id, status)

    // Centralized Operational State
    val operationalState: Flow<OperationalState?> = dao.getOperationalStateFlow()
    suspend fun getOperationalState(): OperationalState? = dao.getOperationalState()
    suspend fun insertOperationalState(state: OperationalState) = dao.insertOperationalState(state)
}
