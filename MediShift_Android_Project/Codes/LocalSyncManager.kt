package com.medishift.ejisay.data

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class SystemLog(
    val id: Int,
    val timestamp: String,
    val category: String,
    val message: String
)

object LocalSyncManager {
    private val _syncStatus = MutableStateFlow("SQLite Room Engine • 100% Offline Local Device")
    val syncStatus: StateFlow<String> = _syncStatus

    private val _realtimeClientsConnected = MutableStateFlow(1)
    val realtimeClientsConnected: StateFlow<Int> = _realtimeClientsConnected

    private val _localEventsLog = MutableStateFlow<List<SystemLog>>(emptyList())
    val localEventsLog: StateFlow<List<SystemLog>> = _localEventsLog

    private var logCounter = 1

    fun initialize(context: Context) {
        logEvent("SYSTEM", "Local Room SQLite database initialized and ready on device.")
    }

    fun logEvent(category: String, message: String) {
        val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        val timeStr = sdf.format(Date())
        val newLog = SystemLog(
            id = logCounter++,
            timestamp = timeStr,
            category = category,
            message = message
        )
        _localEventsLog.value = listOf(newLog) + _localEventsLog.value.take(49)
    }

    fun logEvent(message: String) {
        logEvent("SQLITE", message)
    }

    fun simulateIncomingRosterUpdate(onComplete: (String) -> Unit) {
        logEvent("MUTATION", "Local SQLite trigger: Operational state updated with local roster solver results.")
        onComplete("Local SQLite database roster updated successfully.")
    }

    fun simulateLiveSyncEvent(category: String, message: String) {
        logEvent(category, message)
    }
}

