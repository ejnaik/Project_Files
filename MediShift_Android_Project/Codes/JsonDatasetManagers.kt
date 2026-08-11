package com.medishift.ejisay.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class StaffJsonModel(
    val id: String,
    val name: String,
    val email: String,
    val role: String,
    val seniority: String,
    val status: String,
    val salary: Double,
    val allowances: Double
)

object ShiftDatasetManager {

    private const val FILE_NAME = "shift_dataset_3years.json"

    fun getFile(context: Context): File {
        return File(context.filesDir, FILE_NAME)
    }

    /**
     * Loads the 3-Year Shift Dataset (up to today, 3,288+ records).
     * Reads from local filesDir if available, otherwise initializes from assets.
     */
    fun loadDataset(context: Context): List<ShiftRecord> {
        val file = getFile(context)
        val fromAssets = try {
            context.assets.open(FILE_NAME).bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            ""
        }

        val jsonString: String = if (file.exists()) {
            val content = file.readText()
            if (fromAssets.isNotBlank() && content.length < fromAssets.length / 2) {
                file.writeText(fromAssets)
                fromAssets
            } else {
                content
            }
        } else {
            if (fromAssets.isNotBlank()) {
                file.writeText(fromAssets)
                fromAssets
            } else {
                "[]"
            }
        }

        val records = mutableListOf<ShiftRecord>()
        try {
            val array = JSONArray(jsonString)
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                records.add(
                    ShiftRecord(
                        id = obj.optInt("id", i + 1),
                        date = obj.optString("date", "2026-08-05"),
                        year = obj.optInt("year", 2026),
                        month = obj.optInt("month", 8),
                        dayOfWeek = obj.optString("day_of_week", "Wednesday"),
                        shiftType = obj.optString("shift_type", "Morning"),
                        patientInflow = obj.optInt("patient_inflow", 45),
                        weather = obj.optString("weather", "Normal"),
                        isHoliday = obj.optBoolean("is_holiday", false),
                        isLocalEvent = obj.optBoolean("is_local_event", false)
                    )
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return records
    }

    /**
     * Receptionist action: Update or log shift patient inflow for today or any date.
     * Persists updated dataset to JSON file in local filesDir.
     */
    fun updateOrAddShiftRecord(
        context: Context,
        dateStr: String,
        shiftType: String,
        patientInflow: Int,
        weather: String,
        isHoliday: Boolean,
        isLocalEvent: Boolean
    ): List<ShiftRecord> {
        val currentRecords = loadDataset(context).toMutableList()
        val existingIndex = currentRecords.indexOfFirst { it.date == dateStr && it.shiftType.equals(shiftType, ignoreCase = true) }

        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.US)
        val parsedDate = try { sdf.parse(dateStr) } catch (e: Exception) { Date() }
        val year = SimpleDateFormat("yyyy", Locale.US).format(parsedDate).toIntOrNull() ?: 2026
        val month = SimpleDateFormat("M", Locale.US).format(parsedDate).toIntOrNull() ?: 8
        val dayOfWeek = SimpleDateFormat("EEEE", Locale.US).format(parsedDate)

        if (existingIndex >= 0) {
            val old = currentRecords[existingIndex]
            currentRecords[existingIndex] = old.copy(
                patientInflow = patientInflow,
                weather = weather,
                isHoliday = isHoliday,
                isLocalEvent = isLocalEvent
            )
        } else {
            val newId = (currentRecords.maxOfOrNull { it.id } ?: 0) + 1
            currentRecords.add(
                ShiftRecord(
                    id = newId,
                    date = dateStr,
                    year = year,
                    month = month,
                    dayOfWeek = dayOfWeek,
                    shiftType = shiftType,
                    patientInflow = patientInflow,
                    weather = weather,
                    isHoliday = isHoliday,
                    isLocalEvent = isLocalEvent
                )
            )
        }

        saveDataset(context, currentRecords)
        PythonMLEngine.clearCache()
        return currentRecords
    }

    private fun saveDataset(context: Context, records: List<ShiftRecord>) {
        val jsonArray = JSONArray()
        for (r in records) {
            val obj = JSONObject()
            obj.put("id", r.id)
            obj.put("date", r.date)
            obj.put("year", r.year)
            obj.put("month", r.month)
            obj.put("day_of_week", r.dayOfWeek)
            obj.put("shift_type", r.shiftType)
            obj.put("patient_inflow", r.patientInflow)
            obj.put("weather", r.weather)
            obj.put("is_holiday", r.isHoliday)
            obj.put("is_local_event", r.isLocalEvent)
            // DO NOT include calculated fields (doctors_scheduled, etc.) as required by rule #3
            jsonArray.put(obj)
        }
        getFile(context).writeText(jsonArray.toString(2))
    }
}

object StaffDatasetManager {

    private const val FILE_NAME = "staff_dataset.json"

    fun getFile(context: Context): File {
        return File(context.filesDir, FILE_NAME)
    }

    /**
     * Loads staff dataset from local JSON file (or assets fallback).
     */
    fun loadStaffDataset(context: Context): List<StaffJsonModel> {
        val file = getFile(context)
        val fromAssets = try {
            context.assets.open(FILE_NAME).bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            ""
        }

        val jsonString: String = if (file.exists()) {
            val content = file.readText()
            // If internal file is outdated (e.g. much smaller than asset file), re-sync with assets
            if (fromAssets.isNotBlank() && content.length < fromAssets.length / 2) {
                file.writeText(fromAssets)
                fromAssets
            } else {
                content
            }
        } else {
            if (fromAssets.isNotBlank()) {
                file.writeText(fromAssets)
                fromAssets
            } else {
                "[]"
            }
        }

        val list = mutableListOf<StaffJsonModel>()
        try {
            val array = JSONArray(jsonString)
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                val rawId = if (obj.has("id")) obj.get("id").toString() else "ST${i + 1}"
                list.add(
                    StaffJsonModel(
                        id = rawId,
                        name = obj.optString("name", "Unknown"),
                        email = obj.optString("email", ""),
                        role = obj.optString("role", "Staff"),
                        seniority = obj.optString("seniority", "Junior"),
                        status = obj.optString("status", "Registered"),
                        salary = obj.optDouble("salary", 50000.0),
                        allowances = obj.optDouble("allowances", 10000.0)
                    )
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return list
    }

    /**
     * Saves staff list to JSON file in app local storage.
     */
    fun saveStaffDataset(context: Context, staffList: List<StaffJsonModel>) {
        val jsonArray = JSONArray()
        for (s in staffList) {
            val obj = JSONObject()
            obj.put("id", s.id)
            obj.put("name", s.name)
            obj.put("email", s.email)
            obj.put("role", s.role)
            obj.put("seniority", s.seniority)
            obj.put("status", s.status)
            obj.put("salary", s.salary)
            obj.put("allowances", s.allowances)
            jsonArray.put(obj)
        }
        getFile(context).writeText(jsonArray.toString(2))
    }

    fun addStaff(context: Context, staff: StaffJsonModel): List<StaffJsonModel> {
        val current = loadStaffDataset(context).toMutableList()
        val newId = if (staff.id.isBlank() || staff.id == "0") {
            val prefix = when {
                staff.role.contains("Doctor", ignoreCase = true) -> "DR"
                staff.role.contains("Nurse", ignoreCase = true) -> "NU"
                staff.role.contains("Pharmacist", ignoreCase = true) -> "PH"
                staff.role.contains("Lab", ignoreCase = true) -> "LT"
                else -> "ST"
            }
            "$prefix${String.format(Locale.US, "%03d", current.size + 1)}"
        } else {
            staff.id
        }
        current.add(staff.copy(id = newId))
        saveStaffDataset(context, current)
        return current
    }

    fun deleteStaff(context: Context, staffId: String): List<StaffJsonModel> {
        val current = loadStaffDataset(context).filter { it.id != staffId }
        saveStaffDataset(context, current)
        return current
    }

    fun updateStaff(context: Context, updated: StaffJsonModel): List<StaffJsonModel> {
        val current = loadStaffDataset(context).toMutableList()
        val idx = current.indexOfFirst { it.id == updated.id || it.email.equals(updated.email, ignoreCase = true) }
        if (idx >= 0) {
            current[idx] = updated
        } else {
            current.add(updated)
        }
        saveStaffDataset(context, current)
        return current
    }
}

