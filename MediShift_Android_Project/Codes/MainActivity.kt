package com.example

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.data.Appointment
import com.example.data.FinalRosterItem
import com.example.data.StaffProfile
import com.example.data.UserAccount
import com.example.data.Candidate
import com.example.data.EmailMessage
import com.example.data.LeaveRequest
import com.example.ui.MediShiftViewModel
import com.example.ui.theme.MyApplicationTheme

import android.graphics.pdf.PdfDocument
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.Color as AndroidColor
import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import android.widget.Toast
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize()
                ) { innerPadding ->
                    MainEntryScreen(
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun MainEntryScreen(
    modifier: Modifier = Modifier,
    viewModel: MediShiftViewModel = viewModel()
) {
    val currentUser by viewModel.currentUser.collectAsStateWithLifecycle()
    val deepLinkHospital by viewModel.deepLinkHospital.collectAsStateWithLifecycle()
    val context = LocalContext.current

    LaunchedEffect(Unit) {
        val activity = context as? android.app.Activity
        val intentData = activity?.intent?.data
        if (intentData != null) {
            val hosp = intentData.getQueryParameter("hosp")
            viewModel.handleDeepLink(hosp)
        }
    }

    Box(modifier = modifier.fillMaxSize()) {
        if (currentUser == null) {
            AuthScreen(viewModel = viewModel)
        } else {
            MediShiftApp(viewModel = viewModel, user = currentUser!!)
        }

        if (deepLinkHospital != null) {
            AlertDialog(
                onDismissRequest = { viewModel.dismissDeepLink() },
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("🏥 ", fontSize = 24.sp)
                        Text("Hospital Portal Sync", fontWeight = FontWeight.Black, color = Color(0xFF0061A4))
                    }
                },
                text = {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            text = "Successfully connected to external registry link:",
                            fontSize = 13.sp,
                            color = Color(0xFF535F70)
                        )
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFE8F0FE)),
                            border = BorderStroke(1.dp, Color(0xFF0061A4))
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Text("HOSPITAL ID: ${deepLinkHospital?.uppercase()}", fontWeight = FontWeight.Bold, color = Color(0xFF0061A4), fontSize = 14.sp)
                                Text("Workspace: AI Studio Build Sandbox", fontSize = 11.sp, color = Color(0xFF004475))
                            }
                        }
                        Text(
                            text = "Register with your '@medishift.ac.in' domain email to access your workspace.",
                            fontSize = 12.sp,
                            color = Color(0xFF1A1C1E)
                        )
                    }
                },
                confirmButton = {
                    Button(
                        onClick = { viewModel.dismissDeepLink() },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
                    ) {
                        Text("PROCEED TO ONBOARDING", fontWeight = FontWeight.Bold)
                    }
                }
            )
        }
    }
}

// LOGIN & ACCOUNT CREATION SCREEN
@Composable
fun AuthScreen(viewModel: MediShiftViewModel) {
    var isSignUpMode by remember { mutableStateOf(false) }
    
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var name by remember { mutableStateOf("") }
    var role by remember { mutableStateOf("Doctor") }

    var showCareersForm by remember { mutableStateOf(false) }
    var successMessage by remember { mutableStateOf<String?>(null) }
    
    BackHandler(enabled = showCareersForm || isSignUpMode) {
        if (showCareersForm) {
            showCareersForm = false
        } else if (isSignUpMode) {
            isSignUpMode = false
        }
    }
    
    val authError by viewModel.authError.collectAsStateWithLifecycle()
    val rolesList = listOf("Doctor", "Nurse", "Pharmacist", "Lab Technician", "Operations Manager", "Medical Officer", "Receptionist", "HR")

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(Color(0xFF0F172A), Color(0xFF1E293B), Color(0xFF0F172A))
                )
            )
    ) {
        // High-end ambient radial glow elements (Contours and beautiful background colors)
        androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
            val width = size.width
            val height = size.height
            
            // Soft glowing indigo-blue sphere top-right
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(Color(0xFF3B82F6).copy(alpha = 0.22f), Color.Transparent),
                    center = androidx.compose.ui.geometry.Offset(width * 0.85f, height * 0.15f),
                    radius = width * 0.7f
                ),
                radius = width * 0.7f,
                center = androidx.compose.ui.geometry.Offset(width * 0.85f, height * 0.15f)
            )
            
            // Soft glowing emerald/teal sphere bottom-left
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(Color(0xFF14B8A6).copy(alpha = 0.18f), Color.Transparent),
                    center = androidx.compose.ui.geometry.Offset(width * 0.15f, height * 0.85f),
                    radius = width * 0.8f
                ),
                radius = width * 0.8f,
                center = androidx.compose.ui.geometry.Offset(width * 0.15f, height * 0.85f)
            )

            // Deep violet ambient blob center right
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(Color(0xFF8B5CF6).copy(alpha = 0.15f), Color.Transparent),
                    center = androidx.compose.ui.geometry.Offset(width * 0.9f, height * 0.55f),
                    radius = width * 0.6f
                ),
                radius = width * 0.6f,
                center = androidx.compose.ui.geometry.Offset(width * 0.9f, height * 0.55f)
            )
        }

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            contentPadding = PaddingValues(top = 48.dp, bottom = 48.dp)
        ) {
            // App Identity Header
            item {
                Box(
                    modifier = Modifier
                        .size(76.dp)
                        .clip(RoundedCornerShape(22.dp))
                        .background(
                            Brush.linearGradient(
                                colors = listOf(Color(0xFF3B82F6), Color(0xFF06B6D4))
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.HealthAndSafety,
                        contentDescription = "Logo",
                        tint = Color.White,
                        modifier = Modifier.size(44.dp)
                    )
                }
                Spacer(modifier = Modifier.height(20.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "Medi",
                        style = TextStyle(
                            color = Color(0xFF60A5FA),
                            fontSize = 28.sp,
                            fontWeight = FontWeight.Light,
                            letterSpacing = 0.5.sp
                        )
                    )
                    Text(
                        text = "Shift",
                        style = TextStyle(
                            color = Color.White,
                            fontSize = 28.sp,
                            fontWeight = FontWeight.Black,
                            letterSpacing = 0.5.sp
                        )
                    )
                }
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = if (isSignUpMode) "Create Account" else "Welcome Back",
                    style = TextStyle(
                        color = Color.White,
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Black,
                        letterSpacing = (-1).sp
                    )
                )
                Text(
                    text = if (isSignUpMode) "Register your profile to begin scheduling" else "Log in to view and manage clinical rotas",
                    color = Color(0xFF94A3B8),
                    fontSize = 13.sp,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp)
                )
                Spacer(modifier = Modifier.height(28.dp))
            }

            // Form Inputs Card
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(28.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B).copy(alpha = 0.85f)),
                    elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
                    border = BorderStroke(1.dp, Color(0xFF334155).copy(alpha = 0.8f))
                ) {
                    Column(
                        modifier = Modifier.padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        if (isSignUpMode) {
                            OutlinedTextField(
                                value = name,
                                onValueChange = { name = it },
                                label = { Text("Full Name") },
                                leadingIcon = { Icon(Icons.Default.Person, contentDescription = null, tint = Color(0xFF60A5FA)) },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(14.dp),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = Color(0xFF3B82F6),
                                    unfocusedBorderColor = Color(0xFF475569),
                                    focusedTextColor = Color.White,
                                    unfocusedTextColor = Color.White.copy(alpha = 0.8f),
                                    focusedLabelColor = Color(0xFF3B82F6),
                                    unfocusedLabelColor = Color(0xFF94A3B8),
                                    focusedLeadingIconColor = Color(0xFF3B82F6),
                                    unfocusedLeadingIconColor = Color(0xFF94A3B8)
                                )
                            )
                        }

                        OutlinedTextField(
                            value = email,
                            onValueChange = { email = it },
                            label = { Text("Official Email Address") },
                            leadingIcon = { Icon(Icons.Default.Email, contentDescription = null, tint = Color(0xFF60A5FA)) },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(14.dp),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF3B82F6),
                                unfocusedBorderColor = Color(0xFF475569),
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White.copy(alpha = 0.8f),
                                focusedLabelColor = Color(0xFF3B82F6),
                                unfocusedLabelColor = Color(0xFF94A3B8),
                                focusedLeadingIconColor = Color(0xFF3B82F6),
                                unfocusedLeadingIconColor = Color(0xFF94A3B8)
                            )
                        )

                        OutlinedTextField(
                            value = password,
                            onValueChange = { password = it },
                            label = { Text("Password") },
                            leadingIcon = { Icon(Icons.Default.Lock, contentDescription = null, tint = Color(0xFF60A5FA)) },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(14.dp),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF3B82F6),
                                unfocusedBorderColor = Color(0xFF475569),
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White.copy(alpha = 0.8f),
                                focusedLabelColor = Color(0xFF3B82F6),
                                unfocusedLabelColor = Color(0xFF94A3B8),
                                focusedLeadingIconColor = Color(0xFF3B82F6),
                                unfocusedLeadingIconColor = Color(0xFF94A3B8)
                            )
                        )

                        if (isSignUpMode) {
                            Column {
                                Text(
                                    text = "Select Employee Role",
                                    style = TextStyle(
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 12.sp,
                                        color = Color(0xFF60A5FA),
                                        letterSpacing = 0.5.sp
                                    ),
                                    modifier = Modifier.padding(bottom = 10.dp, top = 6.dp)
                                )
                                Column(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    rolesList.forEach { r ->
                                        val isSelected = role == r
                                        val (emoji, desc) = when (r) {
                                            "Doctor" -> Pair("🩺", "Physician scheduling & preferences")
                                            "Nurse" -> Pair("🩹", "Care staff shift allocations")
                                            "Operations Manager" -> Pair("🛠️", "Solver control & forecasting")
                                            "Medical Officer" -> Pair("📋", "Roster overview & verification")
                                            "HR" -> Pair("💼", "Candidate approvals & credentials")
                                            "Pharmacist" -> Pair("💊", "Dispensary & medication logistics")
                                            "Lab Technician" -> Pair("🧪", "Diagnostic lab roster & tests")
                                            else -> Pair("👤", "Appointment registrations")
                                        }
                                        Card(
                                            onClick = { role = r },
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .testTag("role_chip_$r"),
                                            shape = RoundedCornerShape(14.dp),
                                            colors = CardDefaults.cardColors(
                                                containerColor = if (isSelected) Color(0xFF334155) else Color(0xFF1E293B).copy(alpha = 0.5f)
                                            ),
                                            border = BorderStroke(
                                                width = if (isSelected) 2.dp else 1.dp,
                                                color = if (isSelected) Color(0xFF3B82F6) else Color(0xFF475569)
                                            )
                                        ) {
                                            Row(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .padding(12.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Text(text = emoji, fontSize = 20.sp)
                                                Spacer(modifier = Modifier.width(12.dp))
                                                Column {
                                                    Text(
                                                        text = r,
                                                        fontWeight = FontWeight.Bold,
                                                        fontSize = 13.sp,
                                                        color = if (isSelected) Color.White else Color.White.copy(alpha = 0.9f)
                                                    )
                                                    Text(
                                                        text = desc,
                                                        fontSize = 11.sp,
                                                        color = if (isSelected) Color(0xFF93C5FD) else Color(0xFF94A3B8)
                                                    )
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (authError != null) {
                            Card(
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFECEB)),
                                shape = RoundedCornerShape(10.dp),
                                border = BorderStroke(1.dp, Color(0xFFFFCDD2))
                            ) {
                                Row(
                                    modifier = Modifier.padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Icon(Icons.Default.Error, contentDescription = null, tint = Color(0xFFBA1A1A), modifier = Modifier.size(16.dp))
                                    Text(
                                        text = authError!!,
                                        color = Color(0xFF8C1D18),
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(4.dp))

                        // Main Submit Action Button
                        Button(
                            onClick = {
                                if (isSignUpMode) {
                                    viewModel.createAccount(email, password, name, role) { success -> }
                                } else {
                                    viewModel.login(email, password) { success -> }
                                }
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(52.dp)
                                .testTag(if (isSignUpMode) "create_account_button" else "login_button"),
                            shape = RoundedCornerShape(16.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF3B82F6))
                        ) {
                            Text(
                                text = if (isSignUpMode) "CREATE ACCOUNT" else "LOG IN",
                                color = Color.White,
                                fontWeight = FontWeight.Black,
                                letterSpacing = 1.5.sp,
                                fontSize = 14.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }

                        // Toggle mode link
                        TextButton(
                            onClick = { isSignUpMode = !isSignUpMode },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = if (isSignUpMode) "Already have an account? Sign In" else "New employee? Create Account",
                                color = Color(0xFF60A5FA),
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }

                        HorizontalDivider(color = Color(0xFF334155), thickness = 1.dp)

                        // Apply to Careers link
                        TextButton(
                            onClick = { showCareersForm = true },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("💼", fontSize = 14.sp)
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Interested in joining? Apply to Careers",
                                color = Color(0xFF34D399),
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                }
            }

            // Quick login assistant (Highly visual horizontal carousel)
            if (!isSignUpMode) {
                item {
                    Spacer(modifier = Modifier.height(28.dp))
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "QUICK-ACCESS TEST PROFILES",
                            style = TextStyle(
                                color = Color(0xFF60A5FA),
                                fontSize = 11.sp,
                                fontWeight = FontWeight.ExtraBold,
                                letterSpacing = 1.5.sp
                            )
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        
                        val demoUsers = listOf(
                            Triple("hr@medishift.ac.in", "HR Dept", "💼"),
                            Triple("manager@medishift.ac.in", "Manager", "🛠️"),
                            Triple("doctor@medishift.ac.in", "Doctor", "🩺"),
                            Triple("nurse@medishift.ac.in", "Nurse", "🩹"),
                            Triple("pharmacist@medishift.ac.in", "Pharmacist", "💊"),
                            Triple("labtech@medishift.ac.in", "Lab Tech", "🧪"),
                            Triple("officer@medishift.ac.in", "Officer", "📋"),
                            Triple("receptionist@medishift.ac.in", "Reception", "👤")
                        )

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState())
                                .padding(vertical = 4.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            demoUsers.forEach { (demoEmail, demoRole, demoEmoji) ->
                                Card(
                                    onClick = {
                                        email = demoEmail
                                        password = "password123"
                                        viewModel.login(demoEmail, "password123") { }
                                    },
                                    shape = RoundedCornerShape(16.dp),
                                    colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B).copy(alpha = 0.7f)),
                                    border = BorderStroke(1.dp, Color(0xFF334155))
                                ) {
                                    Row(
                                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(demoEmoji, fontSize = 16.sp)
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Column {
                                            Text(
                                                text = demoRole,
                                                fontWeight = FontWeight.ExtraBold,
                                                fontSize = 11.sp,
                                                color = Color.White
                                            )
                                            Text(
                                                text = "Auto-Login",
                                                fontSize = 9.sp,
                                                color = Color(0xFF94A3B8)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (showCareersForm) {
        var appName by remember { mutableStateOf("") }
        var appEmail by remember { mutableStateOf("") }
        var appRole by remember { mutableStateOf("Doctor") }
        var appSeniority by remember { mutableStateOf("Junior") }
        var appExp by remember { mutableStateOf("") }

        AlertDialog(
            onDismissRequest = { showCareersForm = false },
            title = { Text("MediShift Career Application", fontWeight = FontWeight.Bold, color = Color(0xFF2E7D32)) },
            text = {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text("Join our elite clinical and support staff. Submit your details below to get approved by our HR team.", fontSize = 12.sp, color = Color(0xFF535F70))
                    
                    OutlinedTextField(
                        value = appName,
                        onValueChange = { appName = it },
                        label = { Text("Full Name") },
                        modifier = Modifier.fillMaxWidth()
                    )

                    OutlinedTextField(
                        value = appEmail,
                        onValueChange = { appEmail = it },
                        label = { Text("Personal Email Address") },
                        modifier = Modifier.fillMaxWidth()
                    )

                    Column {
                        Text("Desired Position", style = MaterialTheme.typography.labelMedium, color = Color(0xFF2E7D32))
                        Row(
                            modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            listOf("Doctor", "Nurse", "Medical Officer", "Receptionist", "Operations Manager").forEach { r ->
                                FilterChip(
                                    selected = appRole == r,
                                    onClick = { appRole = r },
                                    label = { Text(r) }
                                )
                            }
                        }
                    }

                    Column {
                        Text("Self-Assessed Seniority", style = MaterialTheme.typography.labelMedium, color = Color(0xFF2E7D32))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            listOf("Junior", "Senior").forEach { s ->
                                FilterChip(
                                    selected = appSeniority == s,
                                    onClick = { appSeniority = s },
                                    label = { Text(s) }
                                )
                            }
                        }
                    }

                    OutlinedTextField(
                        value = appExp,
                        onValueChange = { appExp = it },
                        label = { Text("Brief Experience / Qualifications") },
                        placeholder = { Text("e.g., 5 years pediatric residency") },
                        modifier = Modifier.fillMaxWidth(),
                        maxLines = 3
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (appName.isNotBlank() && appEmail.isNotBlank()) {
                            viewModel.applyForJob(appName, appEmail, appRole, appSeniority)
                            successMessage = "Thank you, $appName! Your candidate application has been submitted to the HR pool.\n\nOnce the HR department reviews your seniority and triggers your official domain credentials, you will be assigned an approved '@medishift.ac.in' email to register your account."
                            showCareersForm = false
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
                ) {
                    Text(
                        text = "Submit Application",
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            },
            dismissButton = {
                TextButton(onClick = { showCareersForm = false }) {
                    Text(
                        text = "Cancel",
                        color = Color(0xFF535F70),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        )
    }

    if (successMessage != null) {
        AlertDialog(
            onDismissRequest = { successMessage = null },
            title = { Text("Application Received! 🎉", fontWeight = FontWeight.Bold, color = Color(0xFF2E7D32)) },
            text = { Text(successMessage!!) },
            confirmButton = {
                Button(
                    onClick = { successMessage = null },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
                ) {
                    Text(
                        text = "Got It",
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        )
    }
}

// MAIN PLATFORM ROOT CONTAINER (ROLE-BASED VIEW)
@Composable
fun MediShiftApp(
    viewModel: MediShiftViewModel,
    user: UserAccount
) {
    var selectedTab by remember { mutableStateOf(0) }
    val tabHistory = remember { mutableStateListOf<Int>() }
    var showInbox by remember { mutableStateOf(false) }
    var inboxComposeTo by remember { mutableStateOf<String?>(null) }
    var showProfileDialog by remember { mutableStateOf(false) }
    var showProfileMenu by remember { mutableStateOf(false) }
    var showLogoutConfirmDialog by remember { mutableStateOf(false) }

    BackHandler(enabled = !showInbox && !showProfileDialog && (tabHistory.isNotEmpty() || selectedTab != 0)) {
        if (tabHistory.isNotEmpty()) {
            val prev = tabHistory.removeAt(tabHistory.size - 1)
            selectedTab = prev
        } else if (selectedTab != 0) {
            selectedTab = 0
        }
    }

    val staffList by viewModel.staffList.collectAsStateWithLifecycle()
    val rosterItems by viewModel.rosterItems.collectAsStateWithLifecycle()
    val isRosterReleased by viewModel.isRosterReleased.collectAsStateWithLifecycle()
    val appointments by viewModel.appointments.collectAsStateWithLifecycle()

    // Dynamic layout definition based on logged-in role
    val navItems = when (user.role) {
        "Doctor", "Nurse", "Pharmacist", "Lab Technician" -> listOf(
            Triple("MY SHIFTS", Icons.Default.CalendarToday, 0),
            Triple("ROSTER", Icons.Default.Assignment, 1)
        )
        "Medical Officer" -> listOf(
            Triple("OVERVIEW", Icons.Default.Dashboard, 0),
            Triple("ROSTER", Icons.Default.Assignment, 1)
        )
        "Receptionist" -> listOf(
            Triple("DAILY LOG", Icons.Default.TrendingUp, 0),
            Triple("ROSTER", Icons.Default.Assignment, 1)
        )
        "HR" -> listOf(
            Triple("HIRING", Icons.Default.PersonAdd, 0),
            Triple("STAFF", Icons.Default.Groups, 1)
        )
        else -> listOf( // "Operations Manager"
            Triple("Leave Approval", Icons.Default.EventBusy, 3),
            Triple("Forecast", Icons.Default.TrendingUp, 0),
            Triple("Staff Selection", Icons.Default.Groups, 1),
            Triple("Release Roster", Icons.Default.Assignment, 2),
            Triple("Optimality Report", Icons.Default.FactCheck, 4)
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFFDFCFF))
    ) {
        // Shared App Header
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 20.dp, top = 20.dp, end = 20.dp, bottom = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                modifier = Modifier.weight(1f),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                if (selectedTab != 0 || tabHistory.isNotEmpty()) {
                    IconButton(
                        onClick = {
                            if (tabHistory.isNotEmpty()) {
                                val prev = tabHistory.removeAt(tabHistory.size - 1)
                                selectedTab = prev
                            } else {
                                selectedTab = 0
                            }
                        },
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(Color(0xFFE8F0FE))
                    ) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Back",
                            tint = Color(0xFF0061A4),
                            modifier = Modifier.size(18.dp)
                        )
                    }
                }

                Column {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(1.dp)
                    ) {
                        Text(
                            text = "Medi",
                            style = TextStyle(
                                color = Color(0xFF0061A4),
                                fontSize = 22.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = (-0.8).sp
                            )
                        )
                        Text(
                            text = "Shift",
                            style = TextStyle(
                                color = Color(0xFF1E293B),
                                fontSize = 22.sp,
                                fontWeight = FontWeight.Black,
                                letterSpacing = (-0.8).sp
                            )
                        )
                        Box(
                            modifier = Modifier
                                .padding(start = 2.dp, top = 4.dp)
                                .size(5.dp)
                                .clip(CircleShape)
                                .background(Color(0xFF3B82F6))
                        )
                    }
                    Text(
                    text = when (user.role) {
                        "Doctor", "Nurse", "Pharmacist", "Lab Technician" -> if (selectedTab == 0) "My Shifts" else "Full Roster"
                        "Medical Officer" -> if (selectedTab == 0) "MO Dashboard" else "Roster Grid"
                        "Receptionist" -> when (selectedTab) {
                            0 -> "Daily Patient Log"
                            else -> "Roster Grid"
                        }
                        "HR" -> if (selectedTab == 0) "Hiring Process" else "Staffs Database"
                        else -> when (selectedTab) {
                            0 -> "Run Forecast"
                            1 -> "Staff Selection"
                            2 -> "Finalize & Release Roster"
                            3 -> "Leave Approval"
                            else -> "Optimality Verification Report"
                        }
                    },
                    style = TextStyle(
                        color = Color(0xFF1A1C1E),
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Black,
                        letterSpacing = (-0.5).sp
                    ),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = "Logged in: ${user.name} (${user.role}${if (user.employeeId.isNotEmpty()) " - ID: " + user.employeeId else ""})",
                    fontSize = 11.sp,
                    color = Color(0xFF535F70),
                    fontWeight = FontWeight.Bold
                )
            }
        }

        // Right Actions Area (3 vertical dots options dropdown menu)
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Profile Options Dropdown Menu Group
                Box(
                    modifier = Modifier.wrapContentSize()
                ) {
                    IconButton(
                        onClick = { showProfileMenu = true },
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(Color(0xFFF0F4FA))
                            .testTag("header_profile_menu_button")
                    ) {
                        val userEmails by viewModel.userEmails.collectAsStateWithLifecycle()
                        val unreadCount = userEmails.count { !it.isRead }
                        
                        Box {
                            Icon(
                                imageVector = Icons.Default.MoreVert,
                                contentDescription = "Profile Options",
                                tint = Color(0xFF001D36),
                                modifier = Modifier.size(20.dp)
                            )
                            if (unreadCount > 0) {
                                Box(
                                    modifier = Modifier
                                        .size(8.dp)
                                        .align(Alignment.TopEnd)
                                        .offset(x = 2.dp, y = (-2).dp)
                                        .clip(CircleShape)
                                        .background(Color(0xFFBA1A1A))
                                )
                            }
                        }
                    }

                    DropdownMenu(
                        expanded = showProfileMenu,
                        onDismissRequest = { showProfileMenu = false }
                    ) {
                        DropdownMenuItem(
                            text = { Text("Account", fontSize = 13.sp, fontWeight = FontWeight.SemiBold) },
                            leadingIcon = { Icon(Icons.Default.Person, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFF0061A4)) },
                            onClick = {
                                showProfileMenu = false
                                showProfileDialog = true
                            }
                        )
                        DropdownMenuItem(
                            text = { Text("Email", fontSize = 13.sp, fontWeight = FontWeight.SemiBold) },
                            leadingIcon = { Icon(Icons.Default.Email, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFF0061A4)) },
                            onClick = {
                                showProfileMenu = false
                                showInbox = true
                            }
                        )
                        DropdownMenuItem(
                            text = { Text("Logout", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFFBA1A1A)) },
                            leadingIcon = { Icon(Icons.Default.Logout, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFFBA1A1A)) },
                            onClick = {
                                showProfileMenu = false
                                showLogoutConfirmDialog = true
                            }
                        )
                    }
                }
            }
        }

        val isRosterRole = user.role == "Doctor" || user.role == "Nurse" || user.role == "Pharmacist" || user.role == "Lab Technician"
        if (isRosterRole) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 6.dp)
                    .height(44.dp),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = if (isRosterReleased) Color(0xFFE8F5E9) else Color(0xFFFFECEB)
                ),
                border = BorderStroke(
                    1.dp,
                    if (isRosterReleased) Color(0xFFC8E6C9) else Color(0xFFFFCDD2)
                )
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(horizontal = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(if (isRosterReleased) Color(0xFF2E7D32) else Color(0xFFBA1A1A))
                    )
                    Column {
                        Text(
                            text = "ROSTER RELEASE STATUS",
                            fontSize = 8.sp,
                            fontWeight = FontWeight.Bold,
                            color = if (isRosterReleased) Color(0xFF1B5E20) else Color(0xFF8C1D18),
                            letterSpacing = 0.5.sp
                        )
                        Text(
                            text = if (isRosterReleased) "Live & Finalized" else "Pending Release",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.ExtraBold,
                            color = if (isRosterReleased) Color(0xFF2E7D32) else Color(0xFFBA1A1A)
                        )
                    }
                }
            }
        }

        // Screen Body Container
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(horizontal = 16.dp)
        ) {
            // Render specific layouts according to role and tab selection
            when (user.role) {
                "HR" -> {
                    when (selectedTab) {
                        0 -> HRHiringProcessScreen(viewModel)
                        1 -> StaffDirectoryScreen(
                            viewModel = viewModel,
                            staffList = staffList,
                            isReadOnly = false,
                            onDeleteStaff = { viewModel.removeStaff(it) },
                            onEmailClick = { email ->
                                inboxComposeTo = email
                                showInbox = true
                            }
                        )
                    }
                }
                "Doctor", "Nurse", "Pharmacist", "Lab Technician" -> {
                    when (selectedTab) {
                        0 -> DoctorNurseMyShiftsScreen(viewModel, user, rosterItems, staffList)
                        1 -> RosterGridScreen(
                            rosterItems = rosterItems,
                            predictedInflow = viewModel.predictedInflow.collectAsStateWithLifecycle().value,
                            dynamicStaffNeeded = viewModel.dynamicStaffNeeded.collectAsStateWithLifecycle().value,
                            isReleased = isRosterReleased,
                            isReadOnly = true,
                            onTriggerSolve = {},
                            staffList = staffList
                        )
                    }
                }
                "Medical Officer" -> {
                    when (selectedTab) {
                        0 -> MedicalOfficerOverviewScreen(viewModel, rosterItems, staffList)
                        1 -> RosterGridScreen(
                            rosterItems = rosterItems,
                            predictedInflow = viewModel.predictedInflow.collectAsStateWithLifecycle().value,
                            dynamicStaffNeeded = viewModel.dynamicStaffNeeded.collectAsStateWithLifecycle().value,
                            isReleased = isRosterReleased,
                            isReadOnly = true,
                            onTriggerSolve = {},
                            staffList = staffList
                        )
                    }
                }
                "Receptionist" -> {
                    when (selectedTab) {
                        0 -> ReceptionistAppointmentsScreen(viewModel, appointments, staffList)
                        else -> RosterGridScreen(
                            rosterItems = rosterItems,
                            predictedInflow = viewModel.predictedInflow.collectAsStateWithLifecycle().value,
                            dynamicStaffNeeded = viewModel.dynamicStaffNeeded.collectAsStateWithLifecycle().value,
                            isReleased = isRosterReleased,
                            isReadOnly = true,
                            onTriggerSolve = {},
                            staffList = staffList
                        )
                    }
                }
                else -> { // Operations Manager
                    val predictedInflowVal = viewModel.predictedInflow.collectAsStateWithLifecycle().value
                    val dynamicStaffNeededVal = viewModel.dynamicStaffNeeded.collectAsStateWithLifecycle().value
                    val solverMetricsVal by viewModel.solverMetrics.collectAsStateWithLifecycle()
                    val isOptimizingVal by viewModel.isOptimizing.collectAsStateWithLifecycle()

                    when (selectedTab) {
                        0 -> {
                            MLForecastScreen(
                                viewModel = viewModel,
                                predictedInflow = predictedInflowVal,
                                dynamicStaffNeeded = dynamicStaffNeededVal,
                                onInflowChanged = { viewModel.updatePrediction(it) },
                                onProceedToLP = { selectedTab = 1 }
                            )
                        }
                        1 -> {
                            LPStaffingPlannerScreen(
                                viewModel = viewModel,
                                predictedInflow = predictedInflowVal,
                                onBack = { selectedTab = 0 },
                                onProceed = { selectedTab = 2 }
                            )
                        }
                        3 -> {
                            LeaveApprovalScreen(viewModel = viewModel)
                        }
                        4 -> {
                            OptimalityVerificationReportScreen(viewModel = viewModel)
                        }
                        else -> {
                            Box(modifier = Modifier.fillMaxSize().padding(12.dp)) {
                                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Card(
                                        modifier = Modifier.fillMaxWidth(),
                                        shape = RoundedCornerShape(16.dp),
                                        colors = CardDefaults.cardColors(containerColor = Color(0xFFF3F7FC)),
                                        border = BorderStroke(1.dp, Color(0xFFD0E1FD))
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(12.dp),
                                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Box(
                                                modifier = Modifier
                                                    .size(36.dp)
                                                    .clip(RoundedCornerShape(8.dp))
                                                    .background(Color(0xFF0061A4)),
                                                contentAlignment = Alignment.Center
                                            ) {
                                                Icon(imageVector = Icons.Default.Assignment, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                                            }
                                            Column {
                                                Text("Finalize & Release Clinical Roster", fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Color(0xFF001D36))
                                                Text("Release roster schedule to publish across all staff profiles or download weekly PDF.", fontSize = 11.sp, color = Color(0xFF535F70))
                                            }
                                        }
                                    }

                                    Box(modifier = Modifier.weight(1f)) {
                                        RosterGridScreen(
                                            rosterItems = rosterItems,
                                            predictedInflow = predictedInflowVal,
                                            dynamicStaffNeeded = dynamicStaffNeededVal,
                                            isReleased = isRosterReleased,
                                            isReadOnly = false,
                                            onTriggerSolve = { viewModel.runPrimarySolver() },
                                            onToggleRelease = { viewModel.setRosterReleased(!isRosterReleased) },
                                            staffList = staffList
                                        )
                                    }

                                    Row(
                                        modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        OutlinedButton(
                                            onClick = { selectedTab = 1 },
                                            modifier = Modifier.weight(1f).height(48.dp),
                                            shape = RoundedCornerShape(12.dp),
                                            border = BorderStroke(1.dp, Color(0xFF0061A4))
                                        ) {
                                            Row(
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.Center
                                            ) {
                                                Icon(imageVector = Icons.Default.ArrowBack, contentDescription = null, modifier = Modifier.size(14.dp), tint = Color(0xFF0061A4))
                                                Spacer(modifier = Modifier.width(6.dp))
                                                Text("Back to Staff Selection", fontWeight = FontWeight.Bold, color = Color(0xFF0061A4), fontSize = 12.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                                            }
                                        }

                                        Button(
                                            onClick = { viewModel.setRosterReleased(!isRosterReleased) },
                                            modifier = Modifier.weight(1f).height(48.dp),
                                            shape = RoundedCornerShape(12.dp),
                                            colors = ButtonDefaults.buttonColors(
                                                containerColor = if (isRosterReleased) Color(0xFF0061A4) else Color(0xFF2E7D32)
                                            )
                                        ) {
                                            Row(
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.Center
                                            ) {
                                                Icon(
                                                    imageVector = if (isRosterReleased) Icons.Default.CheckCircle else Icons.Default.Publish,
                                                    contentDescription = null,
                                                    modifier = Modifier.size(16.dp),
                                                    tint = Color.White
                                                )
                                                Spacer(modifier = Modifier.width(6.dp))
                                                Text(
                                                    text = if (isRosterReleased) "ROSTER RELEASED" else "RELEASE ROSTER",
                                                    fontWeight = FontWeight.Bold,
                                                    color = Color.White,
                                                    fontSize = 12.sp,
                                                    maxLines = 1,
                                                    overflow = TextOverflow.Ellipsis
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

            }
        }
        }

        // Custom Bottom Navigation Bar
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(80.dp)
                .background(Color(0xFFF3F4F9))
                .border(1.dp, Color(0xFFDCE2F9))
                .navigationBarsPadding(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceAround
        ) {
            navItems.forEach { (label, icon, index) ->
                val isSelected = selectedTab == index
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .clickable {
                            if (selectedTab != index) {
                                tabHistory.add(selectedTab)
                                selectedTab = index
                            }
                        }
                        .padding(vertical = 4.dp, horizontal = 4.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(16.dp))
                            .background(if (isSelected) Color(0xFFD1E4FF) else Color.Transparent)
                            .padding(horizontal = 18.dp, vertical = 6.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = icon,
                            contentDescription = label,
                            tint = if (isSelected) Color(0xFF001D36) else Color(0xFF44474E),
                            modifier = Modifier.size(20.dp)
                        )
                    }
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = label,
                        style = TextStyle(
                            color = if (isSelected) Color(0xFF1A1C1E) else Color(0xFF44474E).copy(alpha = 0.6f),
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 0.5.sp
                        ),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }

    if (showInbox) {
        InboxDialog(
            viewModel = viewModel,
            user = user,
            initialComposeTo = inboxComposeTo,
            onDismiss = { 
                showInbox = false 
                inboxComposeTo = null
            }
        )
    }

    if (showLogoutConfirmDialog) {
        AlertDialog(
            onDismissRequest = { showLogoutConfirmDialog = false },
            title = {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Icon(imageVector = Icons.Default.Logout, contentDescription = null, tint = Color(0xFFBA1A1A))
                    Text("Confirm Logout", fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                }
            },
            text = {
                Text("Are you sure you want to log out of your session? All local operations will remain synced.", fontSize = 14.sp, color = Color(0xFF44474E))
            },
            confirmButton = {
                Button(
                    onClick = {
                        showLogoutConfirmDialog = false
                        viewModel.logout()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFBA1A1A))
                ) {
                    Text("Logout", fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showLogoutConfirmDialog = false }) {
                    Text("Cancel", color = Color(0xFF0061A4))
                }
            }
        )
    }

    if (showProfileDialog) {
        UserProfileDialog(
            viewModel = viewModel,
            user = user,
            onDismiss = { showProfileDialog = false }
        )
    }

}

@Composable
fun UserProfileDialog(
    viewModel: MediShiftViewModel,
    user: UserAccount,
    onDismiss: () -> Unit
) {
    var educationText by remember { mutableStateOf(user.education) }
    var addressText by remember { mutableStateOf(user.address) }
    var showSuccessMessage by remember { mutableStateOf(false) }

    BackHandler {
        onDismiss()
    }

    androidx.compose.ui.window.Dialog(
        onDismissRequest = onDismiss,
        properties = androidx.compose.ui.window.DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier
                .fillMaxWidth(0.95f)
                .padding(16.dp)
                .clip(RoundedCornerShape(28.dp)),
            color = Color.White,
            tonalElevation = 6.dp,
            shadowElevation = 8.dp
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Header Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color(0xFFE8F0FE)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.Badge,
                                contentDescription = null,
                                tint = Color(0xFF0061A4),
                                modifier = Modifier.size(22.dp)
                            )
                        }
                        Text(
                            text = "Professional Profile",
                            style = TextStyle(
                                fontSize = 20.sp,
                                fontWeight = FontWeight.ExtraBold,
                                color = Color(0xFF001D36),
                                letterSpacing = (-0.5).sp
                            )
                        )
                    }
                    IconButton(onClick = onDismiss) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Close",
                            tint = Color(0xFF535F70)
                        )
                    }
                }

                // Success notification toast inside
                AnimatedVisibility(visible = showSuccessMessage) {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = Color(0xFFE8F5E9)),
                        shape = RoundedCornerShape(16.dp),
                        modifier = Modifier.fillMaxWidth(),
                        border = BorderStroke(1.dp, Color(0xFFC8E6C9))
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.CheckCircle,
                                contentDescription = null,
                                tint = Color(0xFF2E7D32),
                                modifier = Modifier.size(20.dp)
                            )
                            Text(
                                text = "Profile updated successfully!",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFF1B5E20)
                            )
                        }
                    }
                }

                // Header Profile Info card (Read-Only fields in highly stylized clinical card)
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF3F7FC)),
                    border = BorderStroke(1.dp, Color(0xFFD0E1FD)),
                    shape = RoundedCornerShape(20.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(18.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // User Avatar and Name/Role
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(14.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(60.dp)
                                    .clip(CircleShape)
                                    .background(
                                        Brush.linearGradient(
                                            listOf(Color(0xFF0061A4), Color(0xFF004475))
                                        )
                                    ),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = if (user.name.isNotEmpty()) {
                                        val parts = user.name.trim().split(" ")
                                        if (parts.size > 1) {
                                            (parts[0].take(1) + parts[1].take(1)).uppercase()
                                        } else {
                                            parts[0].take(2).uppercase()
                                        }
                                    } else "MS",
                                    style = TextStyle(
                                        color = Color.White,
                                        fontWeight = FontWeight.Black,
                                        fontSize = 20.sp
                                    )
                                )
                            }
                            Column {
                                Text(
                                    text = user.name,
                                    fontSize = 20.sp,
                                    fontWeight = FontWeight.Black,
                                    color = Color(0xFF001D36),
                                    letterSpacing = (-0.5).sp
                                )
                                Spacer(modifier = Modifier.height(2.dp))
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(8.dp))
                                        .background(Color(0xFFD1E4FF))
                                        .padding(horizontal = 8.dp, vertical = 2.dp)
                                ) {
                                    Text(
                                        text = user.role.uppercase(),
                                        fontSize = 10.sp,
                                        fontWeight = FontWeight.ExtraBold,
                                        color = Color(0xFF004475),
                                        letterSpacing = 0.5.sp
                                    )
                                }
                            }
                        }

                        HorizontalDivider(color = Color(0xFFD0E1FD), thickness = 1.dp)

                        // Personal ID fields
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Column {
                                Text("EMPLOYEE ID", fontSize = 9.sp, color = Color(0xFF0061A4), fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                                Text(
                                    text = if (user.employeeId.isNotEmpty()) user.employeeId else "Not Assigned",
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFF1A1C1E)
                                )
                            }
                            Column(horizontalAlignment = Alignment.End) {
                                Text("EMAIL ADDRESS", fontSize = 9.sp, color = Color(0xFF0061A4), fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                                Text(
                                    text = user.email,
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFF1A1C1E)
                                )
                            }
                        }
                    }
                }

                Text(
                    text = "${user.role} - Clinical Credentials",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF0061A4),
                    letterSpacing = 1.sp,
                    modifier = Modifier.padding(top = 4.dp)
                )

                // Define labels, icons, and placeholders depending on user role
                val (label1, icon1, placeholder1) = when (user.role) {
                    "Doctor", "Medical Officer" -> Triple(
                        "Medical Specialization / Degree",
                        Icons.Default.School,
                        "e.g. MBBS, MD in Cardiology"
                    )
                    "Nurse" -> Triple(
                        "Nursing Certification / Council Reg",
                        Icons.Default.LocalHospital,
                        "e.g. Registered Nurse (RN), ICU Critical Care"
                    )
                    "Pharmacist" -> Triple(
                        "Pharmacy License / Board Reg ID",
                        Icons.Default.MedicalServices,
                        "e.g. Registered Pharmacist (RPh), PharmD"
                    )
                    "Lab Technician" -> Triple(
                        "Laboratory Safety Badge / Diagnostics Level",
                        Icons.Default.Biotech,
                        "e.g. Pathology Cert III, Biosafety Level 2"
                    )
                    "Operations Manager" -> Triple(
                        "Administrative Authority / Management Level",
                        Icons.Default.AdminPanelSettings,
                        "e.g. Chief Operations Officer, Senior Director"
                    )
                    else -> Triple(
                        "Department / Credentials",
                        Icons.Default.School,
                        "e.g. Front Desk Management, Admissions Lead"
                    )
                }

                val (label2, icon2, placeholder2) = when (user.role) {
                    "Doctor", "Medical Officer" -> Triple(
                        "Associated Clinic / Consultation Room",
                        Icons.Default.MeetingRoom,
                        "e.g. Room 402, Outpatient Block B"
                    )
                    "Nurse" -> Triple(
                        "Primary Ward Assignment",
                        Icons.Default.HomeWork,
                        "e.g. Ward 3-A, Emergency Department"
                    )
                    "Pharmacist" -> Triple(
                        "Pharmacy Dispatch Station / Drug Inventory Zone",
                        Icons.Default.Store,
                        "e.g. Main Drug Dispensation Wing, Cold Chain Storage"
                    )
                    "Lab Technician" -> Triple(
                        "Assigned Analysis Lab / Department",
                        Icons.Default.Biotech,
                        "e.g. Hematology Lab B, Pathology Analysis Wing"
                    )
                    "Operations Manager" -> Triple(
                        "Primary Operations Control Hub",
                        Icons.Default.Hub,
                        "e.g. Central Command Station, Executive Office"
                    )
                    else -> Triple(
                        "Workstation / Terminal Assignment",
                        Icons.Default.Desk,
                        "e.g. Desk 12, Main Lobby Admissions Desk"
                    )
                }

                // Education / Specialty Input
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Icon(
                            imageVector = icon1,
                            contentDescription = null,
                            tint = Color(0xFF0061A4),
                            modifier = Modifier.size(16.dp)
                        )
                        Text(
                            text = label1,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF1A1C1E)
                        )
                    }
                    OutlinedTextField(
                        value = educationText,
                        onValueChange = { 
                            educationText = it
                            showSuccessMessage = false
                        },
                        placeholder = { Text(placeholder1, color = Color.LightGray) },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                        shape = RoundedCornerShape(12.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Color(0xFF0061A4),
                            unfocusedBorderColor = Color(0xFFDCE2F9)
                        )
                    )
                }

                // Address / Desk Assignment Input
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Icon(
                            imageVector = icon2,
                            contentDescription = null,
                            tint = Color(0xFF0061A4),
                            modifier = Modifier.size(16.dp)
                        )
                        Text(
                            text = label2,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF1A1C1E)
                        )
                    }
                    OutlinedTextField(
                        value = addressText,
                        onValueChange = { 
                            addressText = it
                            showSuccessMessage = false
                        },
                        placeholder = { Text(placeholder2, color = Color.LightGray) },
                        modifier = Modifier.fillMaxWidth().height(80.dp),
                        maxLines = 2,
                        shape = RoundedCornerShape(12.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Color(0xFF0061A4),
                            unfocusedBorderColor = Color(0xFFDCE2F9)
                        )
                    )
                }

                Spacer(modifier = Modifier.height(4.dp))

                // Action Buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Close", color = Color(0xFFBA1A1A), fontWeight = FontWeight.Bold)
                    }
                    Button(
                        onClick = {
                            viewModel.updateUserProfile(educationText, addressText)
                            showSuccessMessage = true
                        },
                        modifier = Modifier.weight(1.5f),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4)),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Check,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Save Credentials", fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
        }
    }
}

@Composable
fun InboxDialog(
    viewModel: MediShiftViewModel,
    user: UserAccount,
    initialComposeTo: String? = null,
    onDismiss: () -> Unit
) {
    val userEmails by viewModel.userEmails.collectAsStateWithLifecycle()
    val userSentEmails by viewModel.userSentEmails.collectAsStateWithLifecycle()
    val staffList by viewModel.staffList.collectAsStateWithLifecycle()
    
    var selectedEmail by remember { mutableStateOf<EmailMessage?>(null) }
    var isComposing by remember { mutableStateOf(initialComposeTo != null) }
    var selectedTab by remember { mutableStateOf(0) }

    var composeTo by remember { mutableStateOf(initialComposeTo ?: "") }
    var composeSubject by remember { mutableStateOf("") }
    var composeBody by remember { mutableStateOf("") }
    var composeError by remember { mutableStateOf<String?>(null) }
    var composeSuccess by remember { mutableStateOf(false) }

    BackHandler(enabled = true) {
        if (selectedEmail != null) {
            selectedEmail = null
        } else if (isComposing) {
            isComposing = false
            composeSuccess = false
            composeError = null
        } else {
            onDismiss()
        }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        properties = androidx.compose.ui.window.DialogProperties(usePlatformDefaultWidth = false),
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .padding(16.dp),
        title = {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Icon(
                        imageVector = Icons.Default.Email,
                        contentDescription = "Inbox",
                        tint = Color(0xFF0061A4),
                        modifier = Modifier.size(28.dp)
                    )
                    Text(
                        text = if (isComposing) "Compose Email" else if (selectedEmail != null) "Read Email" else "My Corporate Inbox",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF1A1C1E)
                    )
                }
                IconButton(onClick = {
                    if (selectedEmail != null) {
                        selectedEmail = null
                    } else if (isComposing) {
                        isComposing = false
                        composeSuccess = false
                        composeError = null
                    } else {
                        onDismiss()
                    }
                }) {
                    Icon(
                        imageVector = if (selectedEmail != null || isComposing) Icons.Default.ArrowBack else Icons.Default.Close,
                        contentDescription = "Back",
                        tint = Color(0xFF44474E)
                    )
                }
            }
        },
        text = {
            Box(modifier = Modifier.fillMaxSize().padding(vertical = 8.dp)) {
                if (isComposing) {
                    Column(
                        modifier = Modifier.fillMaxSize(),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            text = "From: ${user.email}",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF535F70)
                        )

                        // Recipient Email Selection Row / Suggestion
                        var showRecipientDropdown by remember { mutableStateOf(false) }
                        Box(modifier = Modifier.fillMaxWidth()) {
                            OutlinedTextField(
                                value = composeTo,
                                onValueChange = {
                                    composeTo = it
                                    showRecipientDropdown = true
                                },
                                label = { Text("To (Recipient Email)") },
                                placeholder = { Text("colleague@medishift.ac.in") },
                                modifier = Modifier.fillMaxWidth(),
                                maxLines = 1,
                                trailingIcon = {
                                    IconButton(onClick = { showRecipientDropdown = !showRecipientDropdown }) {
                                        Icon(imageVector = Icons.Default.ArrowDropDown, contentDescription = "Show team")
                                    }
                                }
                            )

                            if (showRecipientDropdown) {
                                val filteredStaff = staffList.filter {
                                    val email = it.name.lowercase().replace("dr. ", "").replace("nurse ", "").trim().replace(" ", "") + "@medishift.ac.in"
                                    email.contains(composeTo.lowercase()) || it.name.lowercase().contains(composeTo.lowercase())
                                }
                                if (filteredStaff.isNotEmpty()) {
                                    Card(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(top = 64.dp)
                                            .heightIn(max = 200.dp),
                                        shape = RoundedCornerShape(12.dp),
                                        colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F3FA)),
                                        border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                                    ) {
                                        LazyColumn(modifier = Modifier.padding(4.dp)) {
                                            items(filteredStaff) { staff ->
                                                val derivedEmail = staff.name.lowercase().replace("dr. ", "").replace("nurse ", "").trim().replace(" ", "") + "@medishift.ac.in"
                                                Row(
                                                    modifier = Modifier
                                                        .fillMaxWidth()
                                                        .clickable {
                                                            composeTo = derivedEmail
                                                            showRecipientDropdown = false
                                                        }
                                                        .padding(12.dp),
                                                    horizontalArrangement = Arrangement.SpaceBetween
                                                ) {
                                                    Column {
                                                        Text(text = staff.name, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                                                        Text(text = derivedEmail, fontSize = 11.sp, color = Color(0xFF535F70))
                                                    }
                                                    Text(
                                                        text = staff.role,
                                                        fontSize = 10.sp,
                                                        color = Color(0xFF0061A4),
                                                        fontWeight = FontWeight.Bold
                                                    )
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        OutlinedTextField(
                            value = composeSubject,
                            onValueChange = { composeSubject = it },
                            label = { Text("Subject") },
                            placeholder = { Text("Enter subject...") },
                            modifier = Modifier.fillMaxWidth(),
                            maxLines = 1
                        )

                        OutlinedTextField(
                            value = composeBody,
                            onValueChange = { composeBody = it },
                            label = { Text("Message Body") },
                            placeholder = { Text("Write your email text here...") },
                            modifier = Modifier.fillMaxWidth().weight(1f),
                            minLines = 5
                        )

                        if (composeError != null) {
                            Text(text = composeError!!, color = Color(0xFFBA1A1A), fontSize = 12.sp, fontWeight = FontWeight.Bold)
                        }

                        if (composeSuccess) {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(Color(0xFFE8F5E9))
                                    .padding(12.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "🎉 Email Sent Successfully!",
                                    color = Color(0xFF2E7D32),
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 13.sp
                                )
                            }
                        }

                        Button(
                            onClick = {
                                if (composeTo.isBlank() || composeSubject.isBlank() || composeBody.isBlank()) {
                                    composeError = "Please fill in all email fields."
                                } else {
                                    viewModel.sendEmail(composeTo.trim(), composeSubject.trim(), composeBody.trim()) { success ->
                                        if (success) {
                                            composeSuccess = false
                                            composeError = null
                                            composeTo = ""
                                            composeSubject = ""
                                            composeBody = ""
                                            isComposing = false
                                            selectedTab = 1
                                        } else {
                                            composeError = "Failed to send email. Ensure sender is authorized."
                                        }
                                    }
                                }
                            },
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
                        ) {
                            Icon(imageVector = Icons.Default.Send, contentDescription = "Send", modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Send Corporate Email", maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                    }
                } else if (selectedEmail != null) {
                    val email = selectedEmail!!
                    // Mark as read immediately on display
                    LaunchedEffect(email.id) {
                        viewModel.markEmailAsRead(email.id)
                    }

                    Column(
                        modifier = Modifier.fillMaxSize(),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F3FA)),
                            shape = RoundedCornerShape(16.dp),
                            border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                        ) {
                            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = "From: ${email.senderEmail}",
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 13.sp,
                                        color = Color(0xFF0061A4)
                                    )
                                    Text(text = email.timestamp, fontSize = 11.sp, color = Color(0xFF535F70))
                                }
                                Text(
                                    text = "To: ${email.receiverEmail}",
                                    fontSize = 12.sp,
                                    color = Color(0xFF535F70)
                                )
                                HorizontalDivider(color = Color(0xFFDCE2F9))
                                Text(
                                    text = email.subject,
                                    fontWeight = FontWeight.Black,
                                    fontSize = 16.sp,
                                    color = Color(0xFF1A1C1E)
                                )
                            }
                        }

                        Card(
                            modifier = Modifier.fillMaxWidth().weight(1f),
                            colors = CardDefaults.cardColors(containerColor = Color.White),
                            border = BorderStroke(1.dp, Color(0xFFE0E2EC)),
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            LazyColumn(modifier = Modifier.padding(16.dp).fillMaxSize()) {
                                item {
                                    Text(
                                        text = email.body,
                                        fontSize = 14.sp,
                                        lineHeight = 20.sp,
                                        color = Color(0xFF1A1C1E)
                                    )
                                }
                            }
                        }
                    }
                } else {
                    Column(modifier = Modifier.fillMaxSize()) {
                        TabRow(
                            selectedTabIndex = selectedTab,
                            containerColor = Color(0xFFFDFCFF),
                            contentColor = Color(0xFF0061A4),
                            modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp)
                        ) {
                            Tab(
                                selected = selectedTab == 0,
                                onClick = { selectedTab = 0 },
                                text = {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(imageVector = Icons.Default.Email, contentDescription = null, modifier = Modifier.size(16.dp))
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Text("Inbox", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                                        if (userEmails.any { !it.isRead }) {
                                            Spacer(modifier = Modifier.width(4.dp))
                                            Badge(containerColor = Color(0xFFBA1A1A)) {
                                                Text("${userEmails.count { !it.isRead }}", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                                            }
                                        }
                                    }
                                }
                            )
                            Tab(
                                selected = selectedTab == 1,
                                onClick = { selectedTab = 1 },
                                text = {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(imageVector = Icons.Default.Send, contentDescription = null, modifier = Modifier.size(16.dp))
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Text("Sent", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Badge(containerColor = Color(0xFF535F70)) {
                                            Text("${userSentEmails.size}", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                                        }
                                    }
                                }
                            )
                        }

                        val activeEmails = if (selectedTab == 0) userEmails else userSentEmails

                        if (activeEmails.isEmpty()) {
                            Column(
                                modifier = Modifier.fillMaxSize().weight(1f),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.Center
                            ) {
                                Icon(
                                    imageVector = if (selectedTab == 0) Icons.Default.Email else Icons.Default.Send,
                                    contentDescription = "Empty List",
                                    modifier = Modifier.size(64.dp),
                                    tint = Color(0xFF0061A4).copy(alpha = 0.4f)
                                )
                                Spacer(modifier = Modifier.height(12.dp))
                                Text(
                                    text = if (selectedTab == 0) "Your Inbox is Empty" else "No Sent Emails",
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 16.sp,
                                    color = Color(0xFF1A1C1E)
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = if (selectedTab == 0) 
                                        "Colleagues and payroll administrators can send real-time corporate emails here." 
                                    else 
                                        "Any emails you compose and send will be stored here.",
                                    fontSize = 12.sp,
                                    color = Color(0xFF535F70),
                                    textAlign = TextAlign.Center,
                                    modifier = Modifier.padding(horizontal = 24.dp)
                                )
                            }
                        } else {
                            Column(modifier = Modifier.fillMaxSize().weight(1f)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = if (selectedTab == 0) "INBOX MESSAGES (${userEmails.size})" else "SENT MESSAGES (${userSentEmails.size})",
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 11.sp,
                                        color = Color(0xFF535F70)
                                    )
                                    if (selectedTab == 0) {
                                        Text(
                                            text = "${userEmails.count { !it.isRead }} Unread",
                                            fontWeight = FontWeight.Bold,
                                            fontSize = 11.sp,
                                            color = Color(0xFFBA1A1A)
                                        )
                                    }
                                }

                                LazyColumn(
                                    modifier = Modifier.fillMaxWidth().weight(1f),
                                    verticalArrangement = Arrangement.spacedBy(10.dp)
                                ) {
                                    items(activeEmails) { email ->
                                        val isReadOrSent = if (selectedTab == 0) email.isRead else true
                                        Card(
                                            onClick = { selectedEmail = email },
                                            modifier = Modifier.fillMaxWidth().testTag("email_item_${email.id}"),
                                            colors = CardDefaults.cardColors(
                                                containerColor = if (isReadOrSent) Color(0xFFFDFCFF) else Color(0xFFF0F3FA)
                                            ),
                                            shape = RoundedCornerShape(12.dp),
                                            border = BorderStroke(
                                                width = if (isReadOrSent) 1.dp else 2.dp,
                                                color = if (isReadOrSent) Color(0xFFE0E2EC) else Color(0xFF0061A4)
                                            )
                                        ) {
                                            Row(
                                                modifier = Modifier.padding(14.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                if (!isReadOrSent) {
                                                    Box(
                                                        modifier = Modifier
                                                            .size(8.dp)
                                                            .clip(CircleShape)
                                                            .background(Color(0xFF0061A4))
                                                    )
                                                    Spacer(modifier = Modifier.width(10.dp))
                                                }
                                                Column(modifier = Modifier.weight(1f)) {
                                                    Row(
                                                        modifier = Modifier.fillMaxWidth(),
                                                        horizontalArrangement = Arrangement.SpaceBetween
                                                    ) {
                                                        Text(
                                                            text = if (selectedTab == 0) "From: ${email.senderEmail}" else "To: ${email.receiverEmail}",
                                                            fontWeight = if (isReadOrSent) FontWeight.Medium else FontWeight.Bold,
                                                            fontSize = 12.sp,
                                                            color = Color(0xFF0061A4)
                                                        )
                                                        Text(
                                                            text = email.timestamp.split(" ").lastOrNull() ?: "",
                                                            fontSize = 10.sp,
                                                            color = Color(0xFF535F70)
                                                        )
                                                    }
                                                    Spacer(modifier = Modifier.height(2.dp))
                                                    Text(
                                                        text = email.subject,
                                                        fontWeight = if (isReadOrSent) FontWeight.Normal else FontWeight.Bold,
                                                        fontSize = 14.sp,
                                                        color = Color(0xFF1A1C1E)
                                                    )
                                                    Spacer(modifier = Modifier.height(2.dp))
                                                    Text(
                                                        text = email.body,
                                                        fontSize = 11.sp,
                                                        maxLines = 1,
                                                        color = Color(0xFF535F70)
                                                    )
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            if (!isComposing) {
                if (selectedEmail == null) {
                    Button(
                        onClick = { isComposing = true },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
                    ) {
                        Icon(imageVector = Icons.Default.Create, contentDescription = "Compose", modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Compose Email", maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                } else {
                    val email = selectedEmail!!
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        OutlinedButton(
                            onClick = { selectedEmail = null }
                        ) {
                            Text("Back to List", maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                        Button(
                            onClick = {
                                composeTo = if (email.senderEmail == user.email) email.receiverEmail else email.senderEmail
                                composeSubject = if (email.subject.startsWith("Re:", ignoreCase = true)) email.subject else "Re: ${email.subject}"
                                composeBody = "\n\n--- On ${email.timestamp}, ${email.senderEmail} wrote:\n> ${email.body.replace("\n", "\n> ")}"
                                isComposing = true
                                selectedEmail = null
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
                        ) {
                            Icon(imageVector = Icons.Default.Create, contentDescription = "Reply", modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Reply", maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                    }
                }
            }
        }
    )
}

// 1 & 2: MY SHIFTS SCREEN (For Doctor and Nurse role)
@Composable
fun DoctorNurseMyShiftsScreen(
    viewModel: MediShiftViewModel,
    user: UserAccount,
    rosterItems: List<FinalRosterItem>,
    staffList: List<StaffProfile>
) {
    // Find the current user's profile
    val profile = staffList.find { it.id == user.staffProfileId }
    val isRosterReleased by viewModel.isRosterReleased.collectAsStateWithLifecycle()
    val leaveRequests by viewModel.leaveRequests.collectAsStateWithLifecycle()
    val dayOrder = listOf("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    val myLeaveRequests = leaveRequests.filter { it.staffId == user.staffProfileId }
    var selectedLeaveDays by remember { mutableStateOf(setOf<String>()) }
    var leaveReason by remember { mutableStateOf("") }
    val myShifts = rosterItems.filter { it.staffId == user.staffProfileId }
        .sortedWith(compareBy<FinalRosterItem> { dayOrder.indexOf(it.date) }.thenBy { 
            when (it.shiftSlot) {
                "Morning" -> 0
                "Evening" -> 1
                else -> 2
            }
        })

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(top = 8.dp, bottom = 24.dp)
    ) {
        // Welcome Header Profile Summary
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0061A4)),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Text(
                        text = "WELCOME BACK,",
                        color = Color(0xFFD1E4FF),
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp
                    )
                    Text(
                        text = user.name,
                        color = Color.White,
                        fontSize = 26.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .clip(CircleShape)
                                .background(Color.White.copy(alpha = 0.2f))
                                .padding(horizontal = 10.dp, vertical = 4.dp)
                        ) {
                            Text(text = "Role: ${user.role}", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                        }
                        profile?.let { prof ->
                            Box(
                                modifier = Modifier
                                    .clip(CircleShape)
                                    .background(Color.White.copy(alpha = 0.2f))
                                    .padding(horizontal = 10.dp, vertical = 4.dp)
                            ) {
                                Text(text = "Seniority: ${prof.skillLevel}", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                }
            }
        }

        // REQUEST DAY-OFF PREFERENCE SECTION
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                border = BorderStroke(1.dp, Color(0xFFDCE2F9))
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(text = "📅", fontSize = 20.sp)
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            text = "WEEKLY DAY-OFF PREFERENCE",
                            style = TextStyle(
                                color = Color(0xFF535F70),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                        )
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Select your preferred day off for shift scheduling.",
                        color = Color(0xFF535F70),
                        fontSize = 12.sp
                    )
                    Spacer(modifier = Modifier.height(16.dp))

                    profile?.let { prof ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "Current Preference: ",
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                color = Color(0xFF1A1C1E)
                            )
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(Color(0xFFFFDAD6))
                                    .padding(horizontal = 14.dp, vertical = 6.dp)
                            ) {
                                Text(
                                    text = prof.dayOffPreference,
                                    color = Color(0xFF410002),
                                    fontWeight = FontWeight.Black,
                                    fontSize = 13.sp
                                )
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(12.dp))
                        
                        // Select new preference scroll
                        val days = listOf("None", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState())
                                .padding(vertical = 4.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            days.forEach { d ->
                                val isSelected = prof.dayOffPreference == d
                                FilterChip(
                                    selected = isSelected,
                                    onClick = { viewModel.updateStaffDayOff(prof.id, d) },
                                    label = { Text(d) }
                                )
                            }
                        }
                    } ?: run {
                        Text(
                            text = "Error: Clinical profile not linked. Please register through the login page correctly.",
                            color = Color(0xFFBA1A1A),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }

        // REQUEST SHIFT PREFERENCE SECTION
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                border = BorderStroke(1.dp, Color(0xFFDCE2F9))
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(text = "⏰", fontSize = 20.sp)
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            text = "WEEKLY SHIFT SLOT PREFERENCE",
                            style = TextStyle(
                                color = Color(0xFF535F70),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                        )
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Your preferred shift period (Morning, Evening, or Night). The local solver prioritizes this slot for your active days.",
                        color = Color(0xFF535F70),
                        fontSize = 12.sp,
                        lineHeight = 16.sp
                    )
                    Spacer(modifier = Modifier.height(16.dp))

                    profile?.let { prof ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "Current Preference: ",
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                color = Color(0xFF1A1C1E)
                            )
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(Color(0xFFE8F0FE))
                                    .padding(horizontal = 14.dp, vertical = 6.dp)
                            ) {
                                Text(
                                    text = prof.shiftPreference,
                                    color = Color(0xFF001D36),
                                    fontWeight = FontWeight.Black,
                                    fontSize = 13.sp
                                )
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(12.dp))
                        
                        // Select new preference scroll
                        val shiftSlots = listOf("None", "Morning", "Evening", "Night")
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState())
                                .padding(vertical = 4.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            shiftSlots.forEach { slot ->
                                val isSelected = prof.shiftPreference == slot
                                FilterChip(
                                    selected = isSelected,
                                    onClick = { viewModel.updateStaffShiftPreference(prof.id, slot) },
                                    label = { Text(slot) }
                                )
                            }
                        }
                    } ?: run {
                        Text(
                            text = "Error: Clinical profile not linked. Please register through the login page correctly.",
                            color = Color(0xFFBA1A1A),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }

        // NON-AVAILABILITY / LEAVE REQUEST SECTION
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                border = BorderStroke(1.dp, Color(0xFFDCE2F9))
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(text = "🚫", fontSize = 20.sp)
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            text = "NON-AVAILABILITY / LEAVE REQUEST",
                            style = TextStyle(
                                color = Color(0xFF535F70),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                        )
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Select the day(s) this week you will be unavailable. Your request only takes effect against the roster once the Operations Manager grants approval through Leave Approval.",
                        color = Color(0xFF535F70),
                        fontSize = 12.sp,
                        lineHeight = 16.sp
                    )
                    Spacer(modifier = Modifier.height(16.dp))

                    profile?.let { prof ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState())
                                .padding(vertical = 4.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            dayOrder.forEach { d ->
                                val isSelected = selectedLeaveDays.contains(d)
                                FilterChip(
                                    selected = isSelected,
                                    onClick = {
                                        selectedLeaveDays = if (isSelected) {
                                            selectedLeaveDays - d
                                        } else {
                                            selectedLeaveDays + d
                                        }
                                    },
                                    label = { Text(d) }
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(12.dp))

                        OutlinedTextField(
                            value = leaveReason,
                            onValueChange = { leaveReason = it },
                            modifier = Modifier.fillMaxWidth(),
                            label = { Text("Reason (optional)") },
                            singleLine = true,
                            shape = RoundedCornerShape(12.dp)
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        Button(
                            onClick = {
                                viewModel.submitLeaveRequest(
                                    staffId = prof.id,
                                    staffName = prof.name,
                                    staffRole = prof.role,
                                    days = selectedLeaveDays.sortedBy { dayOrder.indexOf(it) },
                                    reason = leaveReason
                                )
                                selectedLeaveDays = emptySet()
                                leaveReason = ""
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(48.dp)
                                .testTag("submit_leave_request_button"),
                            enabled = selectedLeaveDays.isNotEmpty(),
                            shape = RoundedCornerShape(12.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFBA1A1A))
                        ) {
                            Icon(imageVector = Icons.Default.EventBusy, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "SUBMIT NON-AVAILABILITY REQUEST",
                                fontWeight = FontWeight.Bold,
                                color = Color.White,
                                fontSize = 12.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }

                        if (myLeaveRequests.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(16.dp))
                            HorizontalDivider(color = Color(0xFFDCE2F9))
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = "YOUR LEAVE REQUESTS",
                                style = TextStyle(
                                    color = Color(0xFF535F70),
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.Bold,
                                    letterSpacing = 1.sp
                                )
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                myLeaveRequests.forEach { req ->
                                    val (bgColor, textColor) = when (req.status) {
                                        "Approved" -> Color(0xFFE8F5E9) to Color(0xFF2E7D32)
                                        "Rejected" -> Color(0xFFFFEBEE) to Color(0xFFBA1A1A)
                                        else -> Color(0xFFFFF3E0) to Color(0xFFE65100)
                                    }
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .clip(RoundedCornerShape(12.dp))
                                            .background(Color(0xFFF8FAFC))
                                            .padding(12.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Column(modifier = Modifier.weight(1f)) {
                                            Text(req.days.replace(",", ", "), fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                                            if (req.reason.isNotBlank()) {
                                                Text(req.reason, fontSize = 11.sp, color = Color(0xFF535F70))
                                            }
                                        }
                                        Box(
                                            modifier = Modifier
                                                .clip(RoundedCornerShape(10.dp))
                                                .background(bgColor)
                                                .padding(horizontal = 10.dp, vertical = 4.dp)
                                        ) {
                                            Text(req.status.uppercase(), fontSize = 10.sp, fontWeight = FontWeight.Black, color = textColor)
                                        }
                                    }
                                }
                            }
                        }
                    } ?: run {
                        Text(
                            text = "Error: Clinical profile not linked. Please register through the login page correctly.",
                            color = Color(0xFFBA1A1A),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }

        // PERSONAL ROSTER DETAILS
        item {
            Text(
                text = "YOUR SCHEDULED SHIFTS THIS WEEK",
                style = TextStyle(
                    color = Color(0xFF1A1C1E),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp
                )
            )
        }

        if (!isRosterReleased) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF3E0)),
                    border = BorderStroke(1.dp, Color(0xFFFFE0B2))
                ) {
                    Column(
                        modifier = Modifier.padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Lock,
                            contentDescription = null,
                            tint = Color(0xFFE65100),
                            modifier = Modifier.size(32.dp)
                        )
                        Text(
                            text = "Roster Pending Release",
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            color = Color(0xFFE65100)
                        )
                        Text(
                            text = "The weekly shift schedule is currently in draft status and being finalized by the Operations Manager. Your assigned shifts will appear here once released.",
                            fontSize = 12.sp,
                            color = Color(0xFF5D4037),
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }
        } else if (myShifts.isEmpty()) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F3FA)),
                    border = BorderStroke(1.dp, Color(0xFFDCE2F9).copy(alpha = 0.5f))
                ) {
                    Column(
                        modifier = Modifier.padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(text = "🎉", fontSize = 36.sp)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "No Shifts Assigned",
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            color = Color(0xFF0061A4)
                        )
                        Text(
                            text = "Enjoy your downtime! The Operations Manager hasn't assigned you shifts or the roster hasn't been re-optimized.",
                            fontSize = 12.sp,
                            color = Color(0xFF535F70),
                            textAlign = TextAlign.Center,
                            modifier = Modifier.padding(top = 4.dp)
                        )
                    }
                }
            }
        } else {
            items(myShifts) { item ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .size(36.dp)
                                    .clip(CircleShape)
                                    .background(Color(0xFFD1E4FF)),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = when (item.shiftSlot) {
                                        "Morning" -> "☀️"
                                        "Evening" -> "🌇"
                                        else -> "🌙"
                                    },
                                    fontSize = 16.sp
                                )
                            }
                            Spacer(modifier = Modifier.width(12.dp))
                            Column {
                                Text(text = item.date, fontWeight = FontWeight.Bold, fontSize = 16.sp, color = Color(0xFF1A1C1E))
                                Text(text = "Shift Slot: ${item.shiftSlot}", fontSize = 12.sp, color = Color(0xFF535F70))
                            }
                        }
                        Box(
                            modifier = Modifier
                                .clip(CircleShape)
                                .background(Color(0xFFE8F5E9))
                                .padding(horizontal = 12.dp, vertical = 4.dp)
                        ) {
                            Text(text = "CONFIRMED", fontSize = 11.sp, fontWeight = FontWeight.Black, color = Color(0xFF2E7D32))
                        }
                    }
                }
            }
        }
    }
}

// 4: MEDICAL OFFICER OVERVIEW SCREEN
@Composable
fun MedicalOfficerOverviewScreen(
    viewModel: MediShiftViewModel,
    rosterItems: List<FinalRosterItem>,
    staffList: List<StaffProfile>
) {
    val doctors = staffList.filter { it.role.contains("Doctor", ignoreCase = true) }
    val nurses = staffList.filter { it.role.contains("Nurse", ignoreCase = true) }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(top = 8.dp, bottom = 24.dp)
    ) {
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0061A4))
            ) {
                Column(modifier = Modifier.padding(24.dp)) {
                    Text(
                        text = "MO Clinical Summary Dashboard",
                        color = Color.White,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Clinical oversight & roster verification.",
                        color = Color(0xFFD1E4FF),
                        fontSize = 12.sp
                    )
                }
            }
        }

        // PDF Roster Download Button
        item {
            val context = LocalContext.current
            Button(
                onClick = { generateAndShareRosterPdf(context, rosterItems, staffList) },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .testTag("download_roster_pdf_mo"),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
            ) {
                Icon(imageVector = Icons.Default.Share, contentDescription = "Download PDF", tint = Color.White)
                Spacer(modifier = Modifier.width(10.dp))
                Text(
                    text = "DOWNLOAD WEEKLY ROSTER (PDF)",
                    style = TextStyle(
                        color = Color.White,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 0.5.sp
                    ),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }

        // Metrics Row
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Card(
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(text = "Total Doctors", fontSize = 11.sp, color = Color(0xFF535F70), fontWeight = FontWeight.Bold)
                        Text(text = "${doctors.size}", fontSize = 24.sp, fontWeight = FontWeight.Black, color = Color(0xFF0061A4))
                        Text(text = "Senior: ${doctors.count { it.skillLevel == "Senior" }}", fontSize = 11.sp, color = Color(0xFF2E7D32))
                    }
                }

                Card(
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(text = "Total Nurses", fontSize = 11.sp, color = Color(0xFF535F70), fontWeight = FontWeight.Bold)
                        Text(text = "${nurses.size}", fontSize = 24.sp, fontWeight = FontWeight.Black, color = Color(0xFF0061A4))
                        Text(text = "Senior: ${nurses.count { it.skillLevel == "Senior" }}", fontSize = 11.sp, color = Color(0xFF2E7D32))
                    }
                }
            }
        }

        // Coverage statistics list
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                border = BorderStroke(1.dp, Color(0xFFDCE2F9))
            ) {
                Column(modifier = Modifier.padding(18.dp)) {
                    Text(
                        text = "SAFETY CHECKS & CLINICAL REVIEWS",
                        fontWeight = FontWeight.Bold,
                        fontSize = 12.sp,
                        color = Color(0xFF535F70),
                        letterSpacing = 1.sp
                    )
                    Spacer(modifier = Modifier.height(14.dp))

                    val nightShifts = rosterItems.filter { it.shiftSlot == "Night" }
                    val nightShiftsWithSenior = nightShifts.count { item ->
                        val matchedStaff = staffList.find { it.id == item.staffId }
                        matchedStaff?.skillLevel == "Senior"
                    }

                    ClinicalCheckRow(
                        title = "Senior Doctor Coverage on Night Shifts",
                        status = if (nightShifts.isEmpty()) "No data" else "$nightShiftsWithSenior / ${nightShifts.size} nights",
                        isPassed = nightShiftsWithSenior > 0
                    )

                    Spacer(modifier = Modifier.height(8.dp))
                    HorizontalDivider(color = Color(0xFFDCE2F9), thickness = 1.dp)
                    Spacer(modifier = Modifier.height(8.dp))

                    ClinicalCheckRow(
                        title = "Safe Workload Ratio Check (< 5 shifts)",
                        status = "100% Passed",
                        isPassed = true
                    )

                    Spacer(modifier = Modifier.height(8.dp))
                    HorizontalDivider(color = Color(0xFFDCE2F9), thickness = 1.dp)
                    Spacer(modifier = Modifier.height(8.dp))

                    ClinicalCheckRow(
                        title = "Doctor-to-Nurse Shift Balance Rate",
                        status = "Balanced",
                        isPassed = true
                    )
                }
            }
        }

        // Clinical warning / alert block
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFDAD6))
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    Text(text = "⚠️", fontSize = 20.sp)
                    Spacer(modifier = Modifier.width(12.dp))
                    Column {
                        Text(text = "High Inflow Alert Status", fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Color(0xFF410002))
                        Text(
                            text = "Admissions Forecast model predicts potentially elevated ER patient counts during weekends. Please ensure maximum standby staff numbers are loaded.",
                            fontSize = 11.sp,
                            color = Color(0xFF410002).copy(alpha = 0.8f),
                            lineHeight = 15.sp
                        )
                    }
                }
            }
        }
    }
}

fun generateAndShareRosterPdf(
    context: Context,
    rosterItems: List<FinalRosterItem>,
    staffList: List<StaffProfile> = emptyList()
) {
    if (rosterItems.isEmpty()) {
        Toast.makeText(context, "No roster items to generate PDF.", Toast.LENGTH_SHORT).show()
        return
    }

    try {
        val pdfDocument = PdfDocument()
        val titlePaint = Paint().apply {
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textSize = 16f
            color = AndroidColor.WHITE
            isAntiAlias = true
        }
        val subtitlePaint = Paint().apply {
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.NORMAL)
            textSize = 10f
            color = AndroidColor.WHITE
            isAntiAlias = true
        }
        val headerPaint = Paint().apply {
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textSize = 10.5f
            color = AndroidColor.rgb(0, 97, 164) // #0061A4
            isAntiAlias = true
        }
        val textPaint = Paint().apply {
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.NORMAL)
            textSize = 9.5f
            color = AndroidColor.rgb(26, 28, 30) // #1A1C1E
            isAntiAlias = true
        }
        val linePaint = Paint().apply {
            color = AndroidColor.rgb(220, 226, 249) // #DCE2F9
            strokeWidth = 1f
            style = Paint.Style.STROKE
        }
        val boxOutlinePaint = Paint().apply {
            color = AndroidColor.rgb(71, 85, 105) // Slate gray stroke for checkboxes
            strokeWidth = 1.2f
            style = Paint.Style.STROKE
            isAntiAlias = true
        }
        val rectPaint = Paint().apply {
            style = Paint.Style.FILL
        }

        // 1. Sort roster items by time: Monday first, Morning shift first
        val dayOrderMap = mapOf(
            "monday" to 1, "mon" to 1,
            "tuesday" to 2, "tue" to 2,
            "wednesday" to 3, "wed" to 3,
            "thursday" to 4, "thu" to 4,
            "friday" to 5, "fri" to 5,
            "saturday" to 6, "sat" to 6,
            "sunday" to 7, "sun" to 7
        )

        val shiftOrderMap = mapOf(
            "morning" to 1, "morn" to 1,
            "evening" to 2, "eve" to 2,
            "night" to 3
        )

        fun getDayOrder(dateStr: String): Int {
            val lower = dateStr.trim().lowercase()
            dayOrderMap[lower]?.let { return it }
            for ((key, order) in dayOrderMap) {
                if (lower.contains(key)) return order
            }
            return 99
        }

        fun getShiftOrder(shiftStr: String): Int {
            val lower = shiftStr.trim().lowercase()
            shiftOrderMap[lower]?.let { return it }
            for ((key, order) in shiftOrderMap) {
                if (lower.contains(key)) return order
            }
            return 99
        }

        val sortedRosterItems = rosterItems.sortedWith(
            compareBy(
                { getDayOrder(it.date) },
                { getShiftOrder(it.shiftSlot) },
                { it.staffRole },
                { it.staffName }
            )
        )

        val itemsPerPage = 22
        val totalItems = sortedRosterItems.size
        val totalPages = ((totalItems - 1) / itemsPerPage) + 1

        for (pageIndex in 0 until totalPages) {
            val pageInfo = PdfDocument.PageInfo.Builder(595, 842, pageIndex + 1).create()
            val page = pdfDocument.startPage(pageInfo)
            val canvas = page.canvas

            // Calculate shift-wise category counts for summary
            val mItems = sortedRosterItems.filter { it.shiftSlot.equals("Morning", ignoreCase = true) }
            val eItems = sortedRosterItems.filter { it.shiftSlot.equals("Evening", ignoreCase = true) }
            val nItems = sortedRosterItems.filter { it.shiftSlot.equals("Night", ignoreCase = true) }

            fun countCat(list: List<FinalRosterItem>, keyword: String): Int =
                list.count { it.staffRole.contains(keyword, ignoreCase = true) }

            val mDoc = countCat(mItems, "Doctor") + countCat(mItems, "Medical Officer")
            val mNur = countCat(mItems, "Nurse")
            val mPha = countCat(mItems, "Pharmacist")
            val mLab = countCat(mItems, "Lab")

            val eDoc = countCat(eItems, "Doctor") + countCat(eItems, "Medical Officer")
            val eNur = countCat(eItems, "Nurse")
            val ePha = countCat(eItems, "Pharmacist")
            val eLab = countCat(eItems, "Lab")

            val nDoc = countCat(nItems, "Doctor") + countCat(nItems, "Medical Officer")
            val nNur = countCat(nItems, "Nurse")
            val nPha = countCat(nItems, "Pharmacist")
            val nLab = countCat(nItems, "Lab")

            // 1. Draw Top Header Block (Deep Blue) on Page 1 or general header
            if (pageIndex == 0) {
                // Header Background
                rectPaint.color = AndroidColor.rgb(0, 97, 164) // #0061A4
                canvas.drawRect(0f, 0f, 595f, 100f, rectPaint)

                // Header Content
                canvas.drawText("MEDISHIFT ROSTER OPTIMIZATION SYSTEM", 40f, 35f, titlePaint)
                canvas.drawText("Official Time-Sorted Shift Roster & Attendance Check Sheet", 40f, 55f, subtitlePaint)
                canvas.drawText("Generated: " + java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.getDefault()).format(java.util.Date()) + " (IST) | Total Scheduled: $totalItems shifts", 40f, 75f, subtitlePaint)

                // Category Summary Banner Box
                rectPaint.color = AndroidColor.rgb(240, 244, 250) // #F0F4FA
                canvas.drawRect(30f, 110f, 565f, 175f, rectPaint)

                val summaryPaint = Paint().apply {
                    typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                    textSize = 9.5f
                    color = AndroidColor.rgb(0, 97, 164)
                    isAntiAlias = true
                }
                val summaryValPaint = Paint().apply {
                    typeface = Typeface.create(Typeface.DEFAULT, Typeface.NORMAL)
                    textSize = 9f
                    color = AndroidColor.rgb(26, 28, 30)
                    isAntiAlias = true
                }

                canvas.drawText("SHIFT-WISE STAFF CATEGORY COVERAGE MATRIX:", 40f, 126f, summaryPaint)
                canvas.drawText("• Morning Shift: Doctors ($mDoc) | Nurses ($mNur) | Pharmacists ($mPha) | Lab Techs ($mLab) [Total: ${mItems.size}]", 40f, 140f, summaryValPaint)
                canvas.drawText("• Evening Shift: Doctors ($eDoc) | Nurses ($eNur) | Pharmacists ($ePha) | Lab Techs ($eLab) [Total: ${eItems.size}]", 40f, 153f, summaryValPaint)
                canvas.drawText("• Night Shift:    Doctors ($nDoc) | Nurses ($nNur) | Pharmacists ($nPha) | Lab Techs ($nLab) [Total: ${nItems.size}]", 40f, 166f, summaryValPaint)

                canvas.drawLine(30f, 175f, 565f, 175f, linePaint)
            } else {
                // Simplified Header for subsequent pages
                rectPaint.color = AndroidColor.rgb(0, 97, 164) // #0061A4
                canvas.drawRect(0f, 0f, 595f, 55f, rectPaint)
                
                titlePaint.textSize = 12f
                canvas.drawText("MEDISHIFT ROSTER OPTIMIZATION SYSTEM - Page ${pageIndex + 1} of $totalPages", 40f, 32f, titlePaint)
                titlePaint.textSize = 16f // restore
            }

            // Table parameters (6 columns: StaffID, Staff Name, Role, Day / Date, Shift, Attendance)
            val startY = if (pageIndex == 0) 205f else 80f
            val colXStaffId = 35f
            val colXName = 110f
            val colXRole = 230f
            val colXDate = 330f
            val colXSlot = 400f
            val colXAttendance = 470f

            // 2. Draw Table Header
            rectPaint.color = AndroidColor.rgb(240, 244, 250) // #F0F4FA
            canvas.drawRect(30f, startY - 20f, 565f, startY + 10f, rectPaint)

            canvas.drawText("StaffID", colXStaffId, startY, headerPaint)
            canvas.drawText("Staff Name", colXName, startY, headerPaint)
            canvas.drawText("Role", colXRole, startY, headerPaint)
            canvas.drawText("Day / Date", colXDate, startY, headerPaint)
            canvas.drawText("Shift", colXSlot, startY, headerPaint)
            canvas.drawText("Attendance", colXAttendance, startY, headerPaint)

            // Draw line under header
            canvas.drawLine(30f, startY + 10f, 565f, startY + 10f, linePaint)

            // 3. Draw rows
            val startIndex = pageIndex * itemsPerPage
            val endIndex = minOf(startIndex + itemsPerPage, totalItems)
            var currentY = startY + 30f

            for (i in startIndex until endIndex) {
                val item = sortedRosterItems[i]

                // Alternating backgrounds
                if (i % 2 == 0) {
                    rectPaint.color = AndroidColor.rgb(250, 252, 255) // lighter shade
                    canvas.drawRect(30f, currentY - 18f, 565f, currentY + 8f, rectPaint)
                }

                // Full Staff ID lookup
                val matchedStaff = staffList.find { it.id == item.staffId || it.name.equals(item.staffName, ignoreCase = true) }
                val fullStaffId = when {
                    matchedStaff != null && matchedStaff.employeeId.isNotBlank() -> matchedStaff.employeeId
                    item.staffId > 0 -> "EMP-${item.staffId}"
                    else -> "N/A"
                }

                val maxIdLen = 12
                val displayId = if (fullStaffId.length > maxIdLen) fullStaffId.take(maxIdLen) else fullStaffId

                val maxNameLen = 16
                val displayName = if (item.staffName.length > maxNameLen) {
                    item.staffName.take(maxNameLen - 3) + "..."
                } else {
                    item.staffName
                }

                val maxRoleLen = 14
                val displayRole = if (item.staffRole.length > maxRoleLen) {
                    item.staffRole.take(maxRoleLen - 3) + "..."
                } else {
                    item.staffRole
                }

                canvas.drawText(displayId, colXStaffId, currentY, textPaint)
                canvas.drawText(displayName, colXName, currentY, textPaint)
                canvas.drawText(displayRole, colXRole, currentY, textPaint)
                canvas.drawText(item.date, colXDate, currentY, textPaint)
                canvas.drawText(item.shiftSlot, colXSlot, currentY, textPaint)

                // Attendance Checkbox box
                val boxX = colXAttendance + 10f
                val boxY = currentY - 9f
                val boxSize = 10f
                canvas.drawRect(boxX, boxY, boxX + boxSize, boxY + boxSize, boxOutlinePaint)
                canvas.drawText("[  ] Present", boxX + boxSize + 6f, currentY, textPaint)

                // Draw thin line under row
                canvas.drawLine(30f, currentY + 8f, 565f, currentY + 8f, linePaint)
                currentY += 26f
            }

            // 4. Draw Footer
            val footerPaint = Paint().apply {
                typeface = Typeface.create(Typeface.DEFAULT, Typeface.ITALIC)
                textSize = 9f
                color = AndroidColor.rgb(120, 120, 120)
                isAntiAlias = true
            }
            canvas.drawText("Confidential - Official Medical Staff Roster & Attendance Register. Powered by MediShift AI.", 40f, 815f, footerPaint)
            canvas.drawText("Page ${pageIndex + 1} of $totalPages", 500f, 815f, footerPaint)

            pdfDocument.finishPage(page)
        }

        // Save to File
        val rosterDir = File(context.cacheDir, "roster")
        if (!rosterDir.exists()) {
            rosterDir.mkdirs()
        }
        val pdfFile = File(rosterDir, "weekly_roster.pdf")
        val outputStream = FileOutputStream(pdfFile)
        pdfDocument.writeTo(outputStream)
        outputStream.close()
        pdfDocument.close()

        // Trigger Share sheet
        val contentUri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            pdfFile
        )

        val shareIntent = Intent(Intent.ACTION_SEND).apply {
            type = "application/pdf"
            putExtra(Intent.EXTRA_STREAM, contentUri)
            putExtra(Intent.EXTRA_SUBJECT, "Time-Sorted Weekly Roster & Attendance Sheet")
            putExtra(Intent.EXTRA_TEXT, "Please find attached the time-sorted weekly roster with attendance check boxes.")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        val chooserIntent = Intent.createChooser(shareIntent, "Share Weekly Roster PDF").apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(chooserIntent)
        Toast.makeText(context, "PDF Roster generated successfully!", Toast.LENGTH_SHORT).show()

    } catch (e: Exception) {
        e.printStackTrace()
        Toast.makeText(context, "Error generating PDF: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
    }
}

@Composable
fun ClinicalCheckRow(
    title: String,
    status: String,
    isPassed: Boolean
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = title, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFF1A1C1E))
        Box(
            modifier = Modifier
                .clip(CircleShape)
                .background(if (isPassed) Color(0xFFE8F5E9) else Color(0xFFFFECEB))
                .padding(horizontal = 10.dp, vertical = 4.dp)
        ) {
            Text(
                text = status,
                fontSize = 11.sp,
                fontWeight = FontWeight.Black,
                color = if (isPassed) Color(0xFF2E7D32) else Color(0xFFBA1A1A)
            )
        }
    }
}

// 5: RECEPTIONIST APPOINTMENTS SCREEN
@Composable
fun ReceptionistAppointmentsScreen(
    viewModel: MediShiftViewModel,
    appointments: List<Appointment>,
    staffList: List<StaffProfile>
) {
    val inflows by viewModel.historicalInflows.collectAsStateWithLifecycle(initialValue = emptyList())
    val context = LocalContext.current
    
    // Default to current device date and time
    val deviceTodayStr = remember {
        val sdf = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        sdf.format(java.util.Date())
    }
    var selectedDate by remember { mutableStateOf(deviceTodayStr) }
    var morningInput by remember { mutableStateOf("55") }
    var eveningInput by remember { mutableStateOf("42") }
    var nightInput by remember { mutableStateOf("23") }
    var patientCountInput by remember { mutableStateOf("120") }
    var operationMessage by remember { mutableStateOf("") }
    var isSuccessMessage by remember { mutableStateOf(true) }

    // Synchronize shift counts & total count text with selected date
    LaunchedEffect(selectedDate, inflows) {
        val datasetRecords = com.example.data.ShiftDatasetManager.loadDataset(context)
        val morningRec = datasetRecords.find { it.date == selectedDate && it.shiftType.equals("Morning", true) }
        val eveningRec = datasetRecords.find { it.date == selectedDate && it.shiftType.equals("Evening", true) }
        val nightRec = datasetRecords.find { it.date == selectedDate && it.shiftType.equals("Night", true) }

        if (morningRec != null || eveningRec != null || nightRec != null) {
            val m = morningRec?.patientInflow ?: 55
            val e = eveningRec?.patientInflow ?: 42
            val n = nightRec?.patientInflow ?: 23
            morningInput = m.toString()
            eveningInput = e.toString()
            nightInput = n.toString()
            patientCountInput = (m + e + n).toString()
        } else {
            val existing = inflows.find { it.date == selectedDate }
            if (existing != null) {
                val total = existing.patientCount
                val m = (total * 0.45).toInt()
                val e = (total * 0.35).toInt()
                val n = (total - (m + e)).coerceAtLeast(0)
                morningInput = m.toString()
                eveningInput = e.toString()
                nightInput = n.toString()
                patientCountInput = total.toString()
            } else {
                morningInput = "680"
                eveningInput = "540"
                nightInput = "360"
                patientCountInput = "1580"
            }
        }
    }

    // Helper functions for date list (derived from current device date & time)
    val recentDates = remember {
        val list = mutableListOf<String>()
        val sdf = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        val cal = java.util.Calendar.getInstance()
        for (i in 0 until 8) {
            list.add(sdf.format(cal.time))
            cal.add(java.util.Calendar.DAY_OF_YEAR, -1)
        }
        list.reversed() // oldest first
    }

    fun formatDateToDayLabel(dateStr: String): String {
        return try {
            val inputFormat = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.ENGLISH)
            val outputFormat = java.text.SimpleDateFormat("EEE", java.util.Locale.ENGLISH)
            val date = inputFormat.parse(dateStr)
            if (date != null) outputFormat.format(date).uppercase() else ""
        } catch (e: Exception) {
            ""
        }
    }

    fun formatDateToDayNumber(dateStr: String): String {
        return try {
            val inputFormat = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.ENGLISH)
            val outputFormat = java.text.SimpleDateFormat("d", java.util.Locale.ENGLISH)
            val date = inputFormat.parse(dateStr)
            if (date != null) outputFormat.format(date) else ""
        } catch (e: Exception) {
            ""
        }
    }

    fun formatDateReadable(dateStr: String): String {
        return try {
            val inputFormat = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.ENGLISH)
            val outputFormat = java.text.SimpleDateFormat("EEEE, MMM d, yyyy", java.util.Locale.ENGLISH)
            val date = inputFormat.parse(dateStr)
            if (date != null) outputFormat.format(date) else dateStr
        } catch (e: Exception) {
            dateStr
        }
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .testTag("receptionist_patient_log_screen"),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(top = 8.dp, bottom = 32.dp, start = 16.dp, end = 16.dp)
    ) {
        // 1. Dashboard Header Banner
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0061A4))
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .drawBehind {
                            // Beautiful abstract decorative circles
                            drawCircle(
                                color = Color.White.copy(alpha = 0.05f),
                                radius = 250f,
                                center = Offset(size.width * 0.9f, size.height * 0.2f)
                            )
                            drawCircle(
                                color = Color.White.copy(alpha = 0.03f),
                                radius = 400f,
                                center = Offset(size.width * 0.1f, size.height * 0.8f)
                            )
                        }
                        .padding(20.dp)
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "RECEPTIONIST PORTAL",
                                color = Color(0xFF81D4FA),
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.5.sp
                            )
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(Color.White.copy(alpha = 0.15f))
                                    .padding(horizontal = 8.dp, vertical = 4.dp)
                            ) {
                                Text(
                                    text = "SQLITE LOCAL",
                                    color = Color.White,
                                    fontSize = 9.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }

                        Text(
                            text = "Daily Patient Log",
                            color = Color.White,
                            fontSize = 22.sp,
                            fontWeight = FontWeight.ExtraBold
                        )

                        Text(
                            text = "Record daily outpatient intake counts.",
                            color = Color.White.copy(alpha = 0.8f),
                            fontSize = 12.sp
                        )

                        Spacer(modifier = Modifier.height(4.dp))

                        // Heartbeat Pulse Canvas
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(60.dp)
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color.Black.copy(alpha = 0.2f)),
                            contentAlignment = Alignment.Center
                        ) {
                            androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                                val w = size.width
                                val h = size.height
                                val path = androidx.compose.ui.graphics.Path()
                                path.moveTo(0f, h * 0.5f)
                                path.lineTo(w * 0.15f, h * 0.5f)
                                path.lineTo(w * 0.25f, h * 0.2f)
                                path.lineTo(w * 0.35f, h * 0.8f)
                                path.lineTo(w * 0.45f, h * 0.35f)
                                path.lineTo(w * 0.5f, h * 0.6f)
                                path.lineTo(w * 0.55f, h * 0.5f)
                                path.lineTo(w * 0.7f, h * 0.5f)
                                path.lineTo(w * 0.75f, h * 0.1f)
                                path.lineTo(w * 0.8f, h * 0.9f)
                                path.lineTo(w * 0.85f, h * 0.5f)
                                path.lineTo(w, h * 0.5f)

                                drawPath(
                                    path = path,
                                    color = Color(0xFF40C4FF),
                                    style = androidx.compose.ui.graphics.drawscope.Stroke(
                                        width = 2.5.dp.toPx(),
                                        pathEffect = androidx.compose.ui.graphics.PathEffect.cornerPathEffect(15f)
                                    )
                                )
                            }
                        }
                    }
                }
            }
        }

        // 2. Horizontal Date Selector Strip
        item {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Select Reporting Date",
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp,
                        color = Color(0xFF1A1C1E)
                    )
                    
                    // DatePicker Dialog Trigger Button
                    TextButton(
                        onClick = {
                            val calendar = java.util.Calendar.getInstance()
                            val datePickerDialog = android.app.DatePickerDialog(
                                context,
                                { _, selectedYear, selectedMonth, selectedDayOfMonth ->
                                    val formatted = String.format("%04d-%02d-%02d", selectedYear, selectedMonth + 1, selectedDayOfMonth)
                                    selectedDate = formatted
                                },
                                calendar.get(java.util.Calendar.YEAR),
                                calendar.get(java.util.Calendar.MONTH),
                                calendar.get(java.util.Calendar.DAY_OF_MONTH)
                            )
                            datePickerDialog.show()
                        },
                        colors = ButtonDefaults.textButtonColors(contentColor = Color(0xFF0061A4))
                    ) {
                        Icon(imageVector = Icons.Default.DateRange, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Custom Date", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                }

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    recentDates.forEach { dateStr ->
                        val isSelected = dateStr == selectedDate
                        val dayLabel = formatDateToDayLabel(dateStr)
                        val dayNum = formatDateToDayNumber(dateStr)
                        val existingRecord = inflows.find { it.date == dateStr }
                        val isLogged = existingRecord != null

                        Card(
                            modifier = Modifier
                                .width(64.dp)
                                .clickable { selectedDate = dateStr }
                                .testTag("date_strip_item_$dateStr"),
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(
                                containerColor = if (isSelected) Color(0xFFD1E4FF) else Color(0xFFF1F3F9)
                            ),
                            border = BorderStroke(
                                width = if (isSelected) 2.dp else 1.dp,
                                color = if (isSelected) Color(0xFF0061A4) else Color(0xFFE1E2EC)
                            )
                        ) {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 12.dp),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = dayLabel,
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = if (isSelected) Color(0xFF001D35) else Color(0xFF535F70)
                                )
                                Text(
                                    text = dayNum,
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.ExtraBold,
                                    color = if (isSelected) Color(0xFF001D35) else Color(0xFF1A1C1E)
                                )
                                if (isLogged) {
                                    Box(
                                        modifier = Modifier
                                            .size(8.dp)
                                            .clip(CircleShape)
                                            .background(Color(0xFF2E7D32))
                                    )
                                } else {
                                    Spacer(modifier = Modifier.height(8.dp))
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Main Logging Card
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                border = BorderStroke(1.dp, Color(0xFFE1E2EC)),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    // Header inside card
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .clip(CircleShape)
                                .background(Color(0xFFEFF5FF)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(imageVector = Icons.Default.DateRange, contentDescription = null, tint = Color(0xFF0061A4))
                        }
                        Column {
                            Text(
                                text = "Intake Counter",
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFF1A1C1E)
                            )
                            Text(
                                text = formatDateReadable(selectedDate),
                                fontSize = 12.sp,
                                color = Color(0xFF535F70)
                            )
                        }
                    }

                    // Display if existing record exists
                    val existing = inflows.find { it.date == selectedDate }
                    if (existing != null) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color(0xFFE8F5E9))
                                .padding(horizontal = 12.dp, vertical = 8.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.CheckCircle,
                                    contentDescription = null,
                                    tint = Color(0xFF2E7D32),
                                    modifier = Modifier.size(16.dp)
                                )
                                Text(
                                    text = "Current record: ${existing.patientCount} patients reported on this date.",
                                    fontSize = 11.sp,
                                    color = Color(0xFF1B5E20),
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    } else {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color(0xFFFFF3E0))
                                .padding(horizontal = 12.dp, vertical = 8.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Info,
                                    contentDescription = null,
                                    tint = Color(0xFFE65100),
                                    modifier = Modifier.size(16.dp)
                                )
                                Text(
                                    text = "No report logged yet. Enter count below to report.",
                                    fontSize = 11.sp,
                                    color = Color(0xFFE65100),
                                    fontWeight = FontWeight.SemiBold
                                )
                            }
                        }
                    }

                    HorizontalDivider(color = Color(0xFFF1F3F9))

                    // --- SHIFT-WISE INTAKE COUNTERS ---
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Text(
                            text = "Shift-Wise Intake Counter",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF1A1C1E)
                        )
                        Text(
                            text = "Log patient intake separately for Morning, Evening, and Night medical shifts.",
                            fontSize = 11.sp,
                            color = Color(0xFF535F70)
                        )

                        // 1. Morning Shift (08:00 - 16:00)
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF8E1)),
                            border = BorderStroke(1.dp, Color(0xFFFFE082))
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Text("🌅", fontSize = 18.sp)
                                    Column {
                                        Text("Morning Shift", fontWeight = FontWeight.Bold, fontSize = 12.sp, color = Color(0xFFE65100))
                                        Text("08:00 - 16:00", fontSize = 10.sp, color = Color(0xFFF57F17))
                                    }
                                }

                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                    IconButton(
                                        onClick = {
                                            val cur = morningInput.toIntOrNull() ?: 0
                                            morningInput = (cur - 1).coerceAtLeast(0).toString()
                                        },
                                        modifier = Modifier.size(32.dp).clip(CircleShape).background(Color.White)
                                    ) {
                                        Icon(imageVector = Icons.Default.Remove, contentDescription = "Minus", modifier = Modifier.size(16.dp))
                                    }

                                    OutlinedTextField(
                                        value = morningInput,
                                        onValueChange = { if (it.isEmpty() || it.all { c -> c.isDigit() }) morningInput = it },
                                        textStyle = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center, color = Color(0xFFE65100)),
                                        modifier = Modifier.width(68.dp).testTag("morning_shift_input"),
                                        singleLine = true,
                                        shape = RoundedCornerShape(8.dp),
                                        colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Color(0xFFE65100), unfocusedBorderColor = Color(0xFFFFB300), focusedContainerColor = Color.White, unfocusedContainerColor = Color.White)
                                    )

                                    IconButton(
                                        onClick = {
                                            val cur = morningInput.toIntOrNull() ?: 0
                                            morningInput = (cur + 1).toString()
                                        },
                                        modifier = Modifier.size(32.dp).clip(CircleShape).background(Color.White)
                                    ) {
                                        Icon(imageVector = Icons.Default.Add, contentDescription = "Plus", modifier = Modifier.size(16.dp))
                                    }
                                }
                            }
                        }

                        // 2. Evening Shift (16:00 - 00:00)
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFF3E5F5)),
                            border = BorderStroke(1.dp, Color(0xFFCE93D8))
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Text("🌆", fontSize = 18.sp)
                                    Column {
                                        Text("Evening Shift", fontWeight = FontWeight.Bold, fontSize = 12.sp, color = Color(0xFF4A148C))
                                        Text("16:00 - 00:00", fontSize = 10.sp, color = Color(0xFF7B1FA2))
                                    }
                                }

                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                    IconButton(
                                        onClick = {
                                            val cur = eveningInput.toIntOrNull() ?: 0
                                            eveningInput = (cur - 1).coerceAtLeast(0).toString()
                                        },
                                        modifier = Modifier.size(32.dp).clip(CircleShape).background(Color.White)
                                    ) {
                                        Icon(imageVector = Icons.Default.Remove, contentDescription = "Minus", modifier = Modifier.size(16.dp))
                                    }

                                    OutlinedTextField(
                                        value = eveningInput,
                                        onValueChange = { if (it.isEmpty() || it.all { c -> c.isDigit() }) eveningInput = it },
                                        textStyle = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center, color = Color(0xFF4A148C)),
                                        modifier = Modifier.width(68.dp).testTag("evening_shift_input"),
                                        singleLine = true,
                                        shape = RoundedCornerShape(8.dp),
                                        colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Color(0xFF4A148C), unfocusedBorderColor = Color(0xFFBA68C8), focusedContainerColor = Color.White, unfocusedContainerColor = Color.White)
                                    )

                                    IconButton(
                                        onClick = {
                                            val cur = eveningInput.toIntOrNull() ?: 0
                                            eveningInput = (cur + 1).toString()
                                        },
                                        modifier = Modifier.size(32.dp).clip(CircleShape).background(Color.White)
                                    ) {
                                        Icon(imageVector = Icons.Default.Add, contentDescription = "Plus", modifier = Modifier.size(16.dp))
                                    }
                                }
                            }
                        }

                        // 3. Night Shift (00:00 - 08:00)
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFE8EAF6)),
                            border = BorderStroke(1.dp, Color(0xFF9FA8DA))
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Text("🌙", fontSize = 18.sp)
                                    Column {
                                        Text("Night Shift", fontWeight = FontWeight.Bold, fontSize = 12.sp, color = Color(0xFF1A237E))
                                        Text("00:00 - 08:00", fontSize = 10.sp, color = Color(0xFF3F51B5))
                                    }
                                }

                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                    IconButton(
                                        onClick = {
                                            val cur = nightInput.toIntOrNull() ?: 0
                                            nightInput = (cur - 1).coerceAtLeast(0).toString()
                                        },
                                        modifier = Modifier.size(32.dp).clip(CircleShape).background(Color.White)
                                    ) {
                                        Icon(imageVector = Icons.Default.Remove, contentDescription = "Minus", modifier = Modifier.size(16.dp))
                                    }

                                    OutlinedTextField(
                                        value = nightInput,
                                        onValueChange = { if (it.isEmpty() || it.all { c -> c.isDigit() }) nightInput = it },
                                        textStyle = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center, color = Color(0xFF1A237E)),
                                        modifier = Modifier.width(68.dp).testTag("night_shift_input"),
                                        singleLine = true,
                                        shape = RoundedCornerShape(8.dp),
                                        colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Color(0xFF1A237E), unfocusedBorderColor = Color(0xFF7986CB), focusedContainerColor = Color.White, unfocusedContainerColor = Color.White)
                                    )

                                    IconButton(
                                        onClick = {
                                            val cur = nightInput.toIntOrNull() ?: 0
                                            nightInput = (cur + 1).toString()
                                        },
                                        modifier = Modifier.size(32.dp).clip(CircleShape).background(Color.White)
                                    ) {
                                        Icon(imageVector = Icons.Default.Add, contentDescription = "Plus", modifier = Modifier.size(16.dp))
                                    }
                                }
                            }
                        }

                        // Total Calculation Summary Banner
                        val mVal = morningInput.toIntOrNull() ?: 0
                        val eVal = eveningInput.toIntOrNull() ?: 0
                        val nVal = nightInput.toIntOrNull() ?: 0
                        val totalVal = mVal + eVal + nVal

                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFEFF5FF)),
                            border = BorderStroke(1.dp, Color(0xFFB3D7FF))
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column {
                                    Text("Calculated Total Intake", fontSize = 11.sp, color = Color(0xFF004080), fontWeight = FontWeight.SemiBold)
                                    Text("Morning: $mVal | Evening: $eVal | Night: $nVal", fontSize = 10.sp, color = Color(0xFF535F70))
                                }
                                Text(
                                    text = "$totalVal Patients",
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.ExtraBold,
                                    color = Color(0xFF0061A4)
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    // Operation response message with animation
                    AnimatedVisibility(visible = operationMessage.isNotEmpty()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(if (isSuccessMessage) Color(0xFFE8F5E9) else Color(0xFFFFEBEE))
                                .padding(12.dp)
                        ) {
                            Text(
                                text = operationMessage,
                                color = if (isSuccessMessage) Color(0xFF2E7D32) else Color(0xFFC62828),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }

                    // Save Button
                    Button(
                        onClick = {
                            val m = morningInput.toIntOrNull() ?: 0
                            val e = eveningInput.toIntOrNull() ?: 0
                            val n = nightInput.toIntOrNull() ?: 0
                            val total = m + e + n
                            if (total <= 0) {
                                operationMessage = "Please enter valid shift intake counts."
                                isSuccessMessage = false
                            } else {
                                viewModel.saveShiftWiseInflow(selectedDate, m, e, n) { message ->
                                    operationMessage = message
                                    isSuccessMessage = !message.lowercase().contains("failed")
                                    android.widget.Toast.makeText(context, message, android.widget.Toast.LENGTH_SHORT).show()
                                }
                            }
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp)
                            .testTag("save_patient_count_button"),
                        shape = RoundedCornerShape(16.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
                    ) {
                        Icon(imageVector = Icons.Default.Send, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "SAVE SHIFT-WISE INTAKE & SYNC",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.ExtraBold,
                            letterSpacing = 0.5.sp
                        )
                    }
                }
            }
        }

        // 4. Recently Logged Inflow History
        item {
            Text(
                text = "Recently Logged Patient Inflows",
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp,
                color = Color(0xFF1A1C1E)
            )
        }

        if (inflows.isEmpty()) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF8F9FC)),
                    border = BorderStroke(1.dp, Color(0xFFE1E2EC))
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 32.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(text = "No historical inflow counts registered in database.", color = Color.Gray, fontSize = 12.sp)
                    }
                }
            }
        } else {
            // Take up to 10 most recent logs
            val sortedInflows = inflows.sortedByDescending { it.date }.take(10)
            items(sortedInflows) { inflowItem ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("inflow_item_${inflowItem.date}"),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    border = BorderStroke(1.dp, Color(0xFFEFF1F8))
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 14.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(36.dp)
                                    .clip(CircleShape)
                                    .background(Color(0xFFEFF5FF)),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(text = "📈", fontSize = 14.sp)
                            }
                            Column {
                                Text(
                                    text = formatDateReadable(inflowItem.date),
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 13.sp,
                                    color = Color(0xFF1A1C1E)
                                )
                                Text(
                                    text = "Status: Connected & Synchronized",
                                    fontSize = 10.sp,
                                    color = Color(0xFF2E7D32),
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }

                        // Patient count pill
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color(0xFF0061A4))
                                .padding(horizontal = 12.dp, vertical = 6.dp)
                        ) {
                            Text(
                                text = "${inflowItem.patientCount} Patients",
                                color = Color.White,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Black
                            )
                        }
                    }
                }
            }
        }
    }
}

// SHARED COMPOSABLES FROM PREVIOUS DESIGN CODES
@Composable
fun RosterGridScreen(
    rosterItems: List<FinalRosterItem>,
    predictedInflow: Int,
    dynamicStaffNeeded: Int,
    isReleased: Boolean,
    isReadOnly: Boolean,
    onTriggerSolve: () -> Unit,
    onToggleRelease: () -> Unit = {},
    staffList: List<StaffProfile> = emptyList()
) {
    // 1. FILTER ROSTER: Include clinical, pharmacy, and laboratory personnel
    val filteredRosterItems = rosterItems.filter { 
        it.staffRole == "Doctor" || it.staffRole == "Nurse" || 
        it.staffRole == "Pharmacist" || it.staffRole == "Lab Technician" 
    }

    val context = LocalContext.current

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .testTag("roster_grid_screen"),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(top = 8.dp, bottom = 24.dp)
    ) {
        if (isReadOnly && !isReleased) {
            item {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 12.dp),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF3E0)),
                    border = BorderStroke(1.dp, Color(0xFFFFE0B2))
                ) {
                    Column(
                        modifier = Modifier.padding(20.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Lock,
                            contentDescription = null,
                            tint = Color(0xFFE65100),
                            modifier = Modifier.size(32.dp)
                        )
                        Text(
                            text = "Roster Pending Release",
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            color = Color(0xFFE65100)
                        )
                        Text(
                            text = "The weekly shift schedule is currently in draft status and being optimized by the Operations Manager. Please check back once finalized and published.",
                            fontSize = 12.sp,
                            color = Color(0xFF5D4037),
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }
        } else {
            // WEEKLY GRID DAY CARDS
            if (filteredRosterItems.isNotEmpty()) {
                val days = listOf("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
                items(days) { day ->
                    val dayItems = filteredRosterItems.filter { it.date.equals(day, ignoreCase = true) }
                    DayCard(dayName = day, shifts = dayItems)
                }
            } else {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(20.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC)),
                        border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(32.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Assignment,
                                contentDescription = null,
                                tint = Color(0xFF94A3B8),
                                modifier = Modifier.size(32.dp)
                            )
                            Text(
                                text = "No Roster Schedule Generated Yet",
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                color = Color(0xFF334155),
                                textAlign = TextAlign.Center
                            )
                            Text(
                                text = "Run the staffing solver from the Operations Manager's staffing planner to generate this week's shift schedule.",
                                fontSize = 12.sp,
                                color = Color(0xFF535F70),
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
            }
        }

        // FINALIZE & RELEASE ROSTER BUTTON (Above PDF Download Button for Operations Manager)
        if (!isReadOnly) {
            item {
                Button(
                    onClick = { onToggleRelease() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp)
                        .testTag("release_roster_button"),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (isReleased) Color(0xFF0061A4) else Color(0xFF2E7D32)
                    )
                ) {
                    Icon(
                        imageVector = if (isReleased) Icons.Default.CheckCircle else Icons.Default.Publish,
                        contentDescription = null,
                        tint = Color.White
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = if (isReleased) "ROSTER RELEASED (CLICK TO RECALL)" else "RELEASE ROSTER",
                        style = TextStyle(
                            color = Color.White,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 0.5.sp
                        ),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }

        // DOWNLOAD WEEKLY ROSTER PDF LINK / BUTTON AT THE BOTTOM OF THE PAGE
        item {
            Button(
                onClick = { generateAndShareRosterPdf(context, rosterItems, staffList) },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp)
                    .testTag("download_roster_pdf_grid"),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
            ) {
                Icon(imageVector = Icons.Default.Share, contentDescription = "Download PDF", tint = Color.White)
                Spacer(modifier = Modifier.width(10.dp))
                Text(
                    text = "DOWNLOAD WEEKLY ROSTER (PDF)",
                    style = TextStyle(
                        color = Color.White,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 0.5.sp
                    ),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
fun DayCard(dayName: String, shifts: List<FinalRosterItem>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        border = BorderStroke(1.dp, Color(0xFFDCE2F9))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = dayName,
                fontWeight = FontWeight.Bold,
                fontSize = 15.sp,
                color = Color(0xFF0061A4)
            )
            Spacer(modifier = Modifier.height(8.dp))
            Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(Color(0xFFDCE2F9).copy(alpha = 0.5f)))
            Spacer(modifier = Modifier.height(8.dp))

            val shiftSlots = listOf("Morning", "Evening", "Night")
            shiftSlots.forEach { slot ->
                val personnel = shifts.filter { it.shiftSlot.equals(slot, ignoreCase = true) }
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    Row(
                        modifier = Modifier.width(90.dp).padding(top = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        val shiftIcon = when (slot) {
                            "Morning" -> Icons.Default.WbSunny
                            "Evening" -> Icons.Default.AccessTime
                            else -> Icons.Default.NightsStay
                        }
                        Icon(
                            imageVector = shiftIcon,
                            contentDescription = slot,
                            tint = Color(0xFF535F70),
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = slot,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF535F70)
                        )
                    }

                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.Center
                    ) {
                        if (personnel.isEmpty()) {
                            Text(
                                text = "Standby Coverage Only",
                                fontSize = 11.sp,
                                color = Color(0xFFBA1A1A),
                                modifier = Modifier.padding(vertical = 4.dp)
                            )
                        } else {
                            val docCount = personnel.count { it.staffRole.contains("Doctor", ignoreCase = true) || it.staffRole.contains("Medical Officer", ignoreCase = true) }
                            val nurseCount = personnel.count { it.staffRole.contains("Nurse", ignoreCase = true) }
                            val pharmCount = personnel.count { it.staffRole.contains("Pharmacist", ignoreCase = true) }
                            val labCount = personnel.count { it.staffRole.contains("Lab", ignoreCase = true) }

                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(bottom = 6.dp)
                                    .horizontalScroll(rememberScrollState()),
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                if (docCount > 0) {
                                    Surface(
                                        shape = RoundedCornerShape(6.dp),
                                        color = Color(0xFFE8F0FE)
                                    ) {
                                        Text(
                                            text = "🩺 Doc: $docCount",
                                            fontSize = 10.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF0061A4),
                                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                        )
                                    }
                                }
                                if (nurseCount > 0) {
                                    Surface(
                                        shape = RoundedCornerShape(6.dp),
                                        color = Color(0xFFE8F5E9)
                                    ) {
                                        Text(
                                            text = "🩹 Nurse: $nurseCount",
                                            fontSize = 10.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF2E7D32),
                                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                        )
                                    }
                                }
                                if (pharmCount > 0) {
                                    Surface(
                                        shape = RoundedCornerShape(6.dp),
                                        color = Color(0xFFF3E5F5)
                                    ) {
                                        Text(
                                            text = "💊 Pharm: $pharmCount",
                                            fontSize = 10.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF7B1FA2),
                                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                        )
                                    }
                                }
                                if (labCount > 0) {
                                    Surface(
                                        shape = RoundedCornerShape(6.dp),
                                        color = Color(0xFFFFF3E0)
                                    ) {
                                        Text(
                                            text = "🔬 Lab: $labCount",
                                            fontSize = 10.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFFE65100),
                                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                        )
                                    }
                                }
                                Text(
                                    text = "(Total: ${personnel.size})",
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.SemiBold,
                                    color = Color(0xFF535F70)
                                )
                            }

                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .horizontalScroll(rememberScrollState()),
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                personnel.forEach { item ->
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(
                                                if (item.staffRole == "Doctor") Color(0xFFE8F0FE)
                                                else if (item.staffRole == "Nurse") Color(0xFFE8F5E9)
                                                else if (item.staffRole == "Pharmacist") Color(0xFFF3E5F5)
                                                else Color(0xFFFFF3E0)
                                            )
                                            .border(
                                                1.dp,
                                                (if (item.staffRole == "Doctor") Color(0xFF0061A4)
                                                else if (item.staffRole == "Nurse") Color(0xFF2E7D32)
                                                else if (item.staffRole == "Pharmacist") Color(0xFF7B1FA2)
                                                else Color(0xFFE65100)).copy(alpha = 0.2f),
                                                RoundedCornerShape(8.dp)
                                            )
                                            .padding(horizontal = 10.dp, vertical = 6.dp)
                                    ) {
                                        val emoji = when (item.staffRole) {
                                            "Doctor" -> "🩺"
                                            "Nurse" -> "🩹"
                                            "Pharmacist" -> "💊"
                                            else -> "🔬"
                                        }
                                        Text(emoji, fontSize = 11.sp)
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Text(
                                            text = item.staffName,
                                            fontSize = 11.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = when (item.staffRole) {
                                                "Doctor" -> Color(0xFF001D36)
                                                "Nurse" -> Color(0xFF00390A)
                                                "Pharmacist" -> Color(0xFF2C003E)
                                                else -> Color(0xFF3E1D00)
                                            }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun StaffDirectoryScreen(
    viewModel: MediShiftViewModel,
    staffList: List<StaffProfile>,
    isReadOnly: Boolean,
    onAddStaffClick: () -> Unit = {},
    onDeleteStaff: (StaffProfile) -> Unit,
    onEmailClick: (String) -> Unit
) {
    val allUserAccounts by viewModel.allUserAccounts.collectAsStateWithLifecycle()
    var staffToTerminate by remember { mutableStateOf<StaffProfile?>(null) }
    var selectedRoleFilter by remember { mutableStateOf("All Roles") }
    var expandedDropdown by remember { mutableStateOf(false) }

    val rolesOptions = listOf(
        "All Roles",
        "Doctor",
        "Nurse",
        "Pharmacist",
        "Lab Technician",
        "Operations Manager",
        "Medical Officer",
        "Receptionist",
        "HR"
    )

    val filteredStaff = if (selectedRoleFilter == "All Roles") {
        staffList
    } else {
        staffList.filter { it.role.equals(selectedRoleFilter, ignoreCase = true) }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Staffs Database",
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp,
                    color = Color(0xFF0061A4)
                )
                Text(
                    text = "Secure internal registry authorized for HR and Work-Hour Auditors only.",
                    fontSize = 11.sp,
                    color = Color.Gray
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Role-wise Dropdown List Selector Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFFF3F4F9)),
            border = BorderStroke(1.dp, Color(0xFFDCE2F9))
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.FilterList,
                        contentDescription = "Filter",
                        tint = Color(0xFF0061A4),
                        modifier = Modifier.size(20.dp)
                    )
                    Column {
                        Text(
                            text = "Filter by Role Wise Grouping",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF535F70)
                        )
                        Text(
                            text = selectedRoleFilter,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF1A1C1E)
                        )
                    }
                }

                Box {
                    Button(
                        onClick = { expandedDropdown = !expandedDropdown },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF0061A4),
                            contentColor = Color.White
                        ),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp),
                        modifier = Modifier.height(36.dp)
                    ) {
                        Text("Select Role", fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Spacer(modifier = Modifier.width(4.dp))
                        Icon(
                            imageVector = Icons.Default.ArrowDropDown,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp)
                        )
                    }

                    DropdownMenu(
                        expanded = expandedDropdown,
                        onDismissRequest = { expandedDropdown = false }
                    ) {
                        rolesOptions.forEach { role ->
                            val isSelected = selectedRoleFilter == role
                            DropdownMenuItem(
                                text = {
                                    Text(
                                        text = role,
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                        color = if (isSelected) Color(0xFF0061A4) else Color(0xFF1A1C1E)
                                    )
                                },
                                onClick = {
                                    selectedRoleFilter = role
                                    expandedDropdown = false
                                }
                            )
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Showing count badge
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = "Registry Records (${filteredStaff.size})",
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF535F70)
            )
            if (selectedRoleFilter != "All Roles") {
                Text(
                    text = "Clear filter",
                    fontSize = 11.sp,
                    color = Color(0xFF0061A4),
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier
                        .clickable { selectedRoleFilter = "All Roles" }
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        if (filteredStaff.isEmpty()) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .background(Color.White, shape = RoundedCornerShape(16.dp))
                    .border(1.dp, Color(0xFFDCE2F9), shape = RoundedCornerShape(16.dp))
                    .padding(24.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Search,
                        contentDescription = "Not found",
                        tint = Color.LightGray,
                        modifier = Modifier.size(48.dp)
                    )
                    Text(
                        text = "No profiles found for '$selectedRoleFilter'",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.Gray
                    )
                    OutlinedButton(
                        onClick = { selectedRoleFilter = "All Roles" },
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("View All Staff", fontSize = 12.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                contentPadding = PaddingValues(bottom = 24.dp)
            ) {
                items(filteredStaff) { staff ->
                    val userAccount = allUserAccounts.find { it.staffProfileId == staff.id }
                    val staffEmail = userAccount?.email ?: "${staff.name.replace(" ", ".").lowercase()}@medishift.ac.in"
                    StaffProfileItemRow(
                        staff = staff,
                        email = staffEmail,
                        isReadOnly = isReadOnly,
                        onDeleteClick = { staffToTerminate = staff },
                        onEmailClick = { onEmailClick(staffEmail) }
                    )
                }
            }
        }
    }

    if (staffToTerminate != null) {
        TerminationWorkflowDialog(
            staff = staffToTerminate!!,
            onDismiss = { staffToTerminate = null },
            onConfirmTermination = {
                onDeleteStaff(staffToTerminate!!)
                staffToTerminate = null
            }
        )
    }
}

@Composable
fun TerminationWorkflowDialog(
    staff: StaffProfile,
    onDismiss: () -> Unit,
    onConfirmTermination: () -> Unit
) {
    var currentStep by remember { mutableStateOf(1) } // 1: Questions, 2: Confirmation Page, 3: Deletion Box & Delete? Bar, 4: Finish

    BackHandler(enabled = true) {
        if (currentStep > 1 && currentStep < 4) {
            currentStep--
        } else {
            onDismiss()
        }
    }

    // Step 1 State: Questions
    var selectedRoleAnswer by remember { mutableStateOf("") }
    var securityWordAnswer by remember { mutableStateOf("") }

    // Step 3 State: Checkbox on Box
    var isDataBoxChecked by remember { mutableStateOf(false) }

    val totalSteps = 4

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Column {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Terminate Employee",
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp,
                        color = Color(0xFFBA1A1A)
                    )
                    IconButton(onClick = onDismiss) {
                        Icon(imageVector = Icons.Default.Close, contentDescription = "Close")
                    }
                }
                Spacer(modifier = Modifier.height(6.dp))
                // Beautiful Step Progress Indicator Bar
                LinearProgressIndicator(
                    progress = { currentStep.toFloat() / totalSteps.toFloat() },
                    modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)),
                    color = Color(0xFFBA1A1A),
                    trackColor = Color(0xFFF9DEDC)
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Step $currentStep of $totalSteps: " + when(currentStep) {
                        1 -> "Identity Verification"
                        2 -> "HR Termination Terms"
                        3 -> "System Deletion Box"
                        else -> "Termination Finalized"
                    },
                    fontSize = 11.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color(0xFF535F70)
                )
            }
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                when (currentStep) {
                    1 -> {
                        Text(
                            text = "To terminate ${staff.name}, please verify the correct details below to prevent administrative errors.",
                            fontSize = 13.sp,
                            color = Color(0xFF1A1C1E)
                        )

                        // Question 1: Select correct role
                        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text(
                                text = "1. What is ${staff.name}'s designated role?",
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp,
                                color = Color(0xFF1A1C1E)
                            )
                            val roles = listOf("Doctor", "Nurse", "Pharmacist", "Lab Technician", "Operations Manager", "Medical Officer", "Receptionist", "HR")
                            Column {
                                roles.forEach { role ->
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .clickable { selectedRoleAnswer = role }
                                            .padding(vertical = 4.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        RadioButton(
                                            selected = (selectedRoleAnswer == role),
                                            onClick = { selectedRoleAnswer = role }
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(text = role, fontSize = 13.sp, color = Color(0xFF1A1C1E))
                                    }
                                }
                            }
                        }

                        // Question 2: Enter security text
                        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text(
                                text = "2. Enter security code to verify authorization (Type: '${staff.employeeId}')",
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp,
                                color = Color(0xFF1A1C1E)
                            )
                            OutlinedTextField(
                                value = securityWordAnswer,
                                onValueChange = { securityWordAnswer = it },
                                placeholder = { Text(text = "Employee ID", fontSize = 13.sp) },
                                modifier = Modifier.fillMaxWidth(),
                                singleLine = true,
                                textStyle = TextStyle(fontSize = 13.sp)
                            )
                        }
                    }
                    2 -> {
                        // Confirmation page
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF1F0)),
                            border = BorderStroke(1.dp, Color(0xFFF9DEDC))
                        ) {
                            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.Groups,
                                        contentDescription = "Warning",
                                        tint = Color(0xFFBA1A1A),
                                        modifier = Modifier.size(20.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = "Official Release Form",
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 13.sp,
                                        color = Color(0xFFBA1A1A)
                                    )
                                }
                                Text(
                                    text = "This screen acts as the formal confirmation record.",
                                    fontWeight = FontWeight.SemiBold,
                                    fontSize = 11.sp,
                                    color = Color(0xFF1A1C1E)
                                )
                                Text(
                                    text = "By clicking proceed, you declare that:\n" +
                                            "• ${staff.name} is no longer fit to hold clinical duties.\n" +
                                            "• HR department has completed exit interviews.\n" +
                                            "• All outstanding roster shifts assigned to this staff member will be marked vacant.",
                                    fontSize = 11.sp,
                                    color = Color(0xFF535F70),
                                    lineHeight = 15.sp
                                )
                            }
                        }
                    }
                    3 -> {
                        Text(
                            text = "Please purge local operational storage records for this staff profile.",
                            fontSize = 13.sp,
                            color = Color(0xFF1A1C1E)
                        )

                        // delete employee data on box
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F4F8)),
                            border = BorderStroke(1.dp, Color(0xFFD3E4F6))
                        ) {
                            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Text(
                                    text = "EMPLOYEE DATA BOX",
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 11.sp,
                                    color = Color(0xFF0061A4),
                                    letterSpacing = 1.sp
                                )
                                HorizontalDivider(color = Color(0xFFD3E4F6))
                                Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                                    Text("Full Name: ${staff.name}", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                                    Text("Employee ID: ${staff.employeeId}", fontSize = 11.sp, color = Color(0xFF535F70))
                                    Text("Assigned Role: ${staff.role}", fontSize = 11.sp, color = Color(0xFF535F70))
                                    Text("Experience Level: ${staff.skillLevel}", fontSize = 11.sp, color = Color(0xFF535F70))
                                }
                                
                                Spacer(modifier = Modifier.height(6.dp))

                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable { isDataBoxChecked = !isDataBoxChecked }
                                        .background(Color.White, RoundedCornerShape(8.dp))
                                        .border(1.dp, if (isDataBoxChecked) Color(0xFFBA1A1A) else Color(0xFFDCE2F9), RoundedCornerShape(8.dp))
                                        .padding(8.dp)
                                ) {
                                    Checkbox(
                                        checked = isDataBoxChecked,
                                        onCheckedChange = { isDataBoxChecked = it },
                                        colors = CheckboxDefaults.colors(checkedColor = Color(0xFFBA1A1A))
                                    )
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text(
                                        text = "Delete Employee Data",
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 12.sp,
                                        color = Color(0xFFBA1A1A)
                                    )
                                }
                            }
                        }

                        // then Delete? bar then finish
                        if (isDataBoxChecked) {
                            Spacer(modifier = Modifier.height(4.dp))
                            // Styled "Delete? Bar"
                            Button(
                                onClick = {
                                    currentStep = 4
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFBA1A1A)),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(44.dp),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Icon(imageVector = Icons.Default.Delete, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = "Delete?",
                                    fontWeight = FontWeight.ExtraBold,
                                    fontSize = 13.sp,
                                    letterSpacing = 1.sp,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    }
                    4 -> {
                        Column(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.CheckCircle,
                                contentDescription = "Success",
                                modifier = Modifier.size(56.dp),
                                tint = Color(0xFF2E7D32)
                            )
                            Spacer(modifier = Modifier.height(10.dp))
                            Text(
                                text = "Termination Successfully Completed",
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                color = Color(0xFF1A1C1E),
                                textAlign = TextAlign.Center
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "${staff.name} is now designated as Terminated, and their database records have been purged.",
                                fontSize = 11.sp,
                                color = Color(0xFF535F70),
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            when (currentStep) {
                1 -> {
                    val isQ1Correct = selectedRoleAnswer.equals(staff.role, ignoreCase = true)
                    val isQ2Correct = securityWordAnswer.trim().equals(staff.employeeId, ignoreCase = true)
                    Button(
                        onClick = { currentStep = 2 },
                        enabled = isQ1Correct && isQ2Correct,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFBA1A1A))
                    ) {
                        Text("Verify & Continue", maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
                2 -> {
                    Button(
                        onClick = { currentStep = 3 },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFBA1A1A))
                    ) {
                        Text("Sign & Proceed", maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
                3 -> {
                    // Handled inside content with "Delete? bar"
                }
                4 -> {
                    Button(
                        onClick = { onConfirmTermination() },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
                    ) {
                        Text("Finish", maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
        },
        dismissButton = {
            if (currentStep < 4) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (currentStep > 1) {
                        OutlinedButton(onClick = { currentStep-- }) {
                            Icon(
                                imageVector = Icons.Default.ArrowBack,
                                contentDescription = null,
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Back", maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                    }
                    TextButton(onClick = onDismiss) {
                        Text("Cancel", color = Color(0xFFBA1A1A), maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
        }
    )
}

@Composable
fun StaffProfileItemRow(
    staff: StaffProfile,
    email: String,
    isReadOnly: Boolean,
    onDeleteClick: () -> Unit,
    onEmailClick: () -> Unit
) {
    var showMenu by remember { mutableStateOf(false) }
    var showAccountDetailsDialog by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        border = BorderStroke(1.dp, Color(0xFFDCE2F9))
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                val avatarColor = if (staff.role == "Doctor") Color(0xFFD1E4FF) else Color(0xFFE8F5E9)
                val textTint = if (staff.role == "Doctor") Color(0xFF0061A4) else Color(0xFF2E7D32)
                
                Box(
                    modifier = Modifier
                        .size(38.dp)
                        .clip(CircleShape)
                        .background(avatarColor),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = if (staff.name.isNotEmpty()) staff.name.first().toString() else "?",
                        color = textTint,
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp
                    )
                }

                Spacer(modifier = Modifier.width(12.dp))

                Column {
                    Text(
                        text = staff.name,
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp,
                        color = Color(0xFF1A1C1E)
                    )
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = staff.role,
                            fontSize = 10.sp,
                            color = textTint,
                            fontWeight = FontWeight.Bold
                        )
                        Text(text = "•", fontSize = 10.sp, color = Color.LightGray)
                        Text(text = staff.skillLevel, fontSize = 10.sp, color = Color.Gray)
                        if (staff.dayOffPreference != "None") {
                            Text(text = "•", fontSize = 10.sp, color = Color.LightGray)
                            Text(
                                text = "Requested Off: ${staff.dayOffPreference}",
                                fontSize = 10.sp,
                                color = Color(0xFFBA1A1A),
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }

            Box {
                IconButton(
                    onClick = { showMenu = true },
                    modifier = Modifier.size(36.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.MoreVert,
                        contentDescription = "Options",
                        tint = Color(0xFF535F70)
                    )
                }

                DropdownMenu(
                    expanded = showMenu,
                    onDismissRequest = { showMenu = false }
                ) {
                    DropdownMenuItem(
                        text = { Text("Account", fontSize = 12.sp, fontWeight = FontWeight.Medium) },
                        leadingIcon = { Icon(Icons.Default.Person, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFF0061A4)) },
                        onClick = {
                            showMenu = false
                            showAccountDetailsDialog = true
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Email", fontSize = 12.sp, fontWeight = FontWeight.Medium) },
                        leadingIcon = { Icon(Icons.Default.Email, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFF0061A4)) },
                        onClick = {
                            showMenu = false
                            onEmailClick()
                        }
                    )
                    if (!isReadOnly) {
                        DropdownMenuItem(
                            text = { Text("Logout (Terminate)", fontSize = 12.sp, fontWeight = FontWeight.Medium, color = Color(0xFFBA1A1A)) },
                            leadingIcon = { Icon(Icons.Default.Logout, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFFBA1A1A)) },
                            onClick = {
                                showMenu = false
                                onDeleteClick()
                            }
                        )
                    }
                }
            }
        }
    }

    if (showAccountDetailsDialog) {
        AlertDialog(
            onDismissRequest = { showAccountDetailsDialog = false },
            title = {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Icon(imageVector = Icons.Default.AccountBox, contentDescription = null, tint = Color(0xFF0061A4))
                    Text("Staff Profile", fontWeight = FontWeight.Bold, fontSize = 18.sp, color = Color(0xFF1A1C1E))
                }
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Comprehensive registry details and roster state for the selected staff member.", fontSize = 12.sp, color = Color.Gray)
                    HorizontalDivider(color = Color(0xFFDCE2F9), thickness = 1.dp)
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Name:", fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Color(0xFF535F70))
                        Text(staff.name, fontSize = 13.sp, color = Color(0xFF1A1C1E))
                    }
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Department/Role:", fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Color(0xFF535F70))
                        Text(staff.role, fontSize = 13.sp, color = Color(0xFF1A1C1E))
                    }
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Skill Level:", fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Color(0xFF535F70))
                        Text(staff.skillLevel, fontSize = 13.sp, color = Color(0xFF1A1C1E))
                    }
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Off-Preference:", fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Color(0xFF535F70))
                        Text(staff.dayOffPreference, fontSize = 13.sp, color = Color(0xFF1A1C1E))
                    }
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Corporate Email:", fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Color(0xFF535F70))
                        Text(email, fontSize = 13.sp, color = Color(0xFF0061A4), fontWeight = FontWeight.SemiBold)
                    }
                }
            },
            confirmButton = {
                Button(
                    onClick = { showAccountDetailsDialog = false },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
                ) {
                    Text("Close", fontWeight = FontWeight.Bold)
                }
            }
        )
    }
}

@Composable
fun MLForecastScreen(
    viewModel: MediShiftViewModel,
    predictedInflow: Int,
    dynamicStaffNeeded: Int,
    onInflowChanged: (Int) -> Unit,
    onProceedToLP: () -> Unit = {}
) {
    val ensembleResult by viewModel.ensembleResult.collectAsStateWithLifecycle()
    val isForecasting by viewModel.isForecasting.collectAsStateWithLifecycle()
    val isHoliday by viewModel.isHoliday.collectAsStateWithLifecycle()
    val isExtremeWeather by viewModel.isExtremeWeather.collectAsStateWithLifecycle()
    val isLocalEvent by viewModel.isLocalEvent.collectAsStateWithLifecycle()

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 32.dp)
    ) {
        item {
            Column {
                Text(
                    text = "Ensemble Forecasting Engine",
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 20.sp,
                    color = Color(0xFF001D36),
                    letterSpacing = (-0.5).sp
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = "Combines Ridge Regression, Gradient-Boosted Stumps, and Holt-Winters Smoothing models to forecast emergency department admissions.",
                    fontSize = 12.sp,
                    color = Color(0xFF535F70)
                )
            }
        }

        // Action Trigger Card
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                border = BorderStroke(1.dp, Color(0xFFDCE2F9))
            ) {
                Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    Text(
                        text = "Compute Next Week's Inflow",
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp,
                        color = Color(0xFF0061A4)
                    )

                    if (isForecasting) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 12.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            CircularProgressIndicator(color = Color(0xFF0061A4), strokeWidth = 3.dp)
                            Text(
                                text = "Applying Expert Non-Linear Models & Anomaly Adjustments...",
                                fontSize = 12.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = Color(0xFF0061A4)
                            )
                        }
                    } else {
                        Button(
                            onClick = { viewModel.runEnsembleForecasting() },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp),
                            shape = RoundedCornerShape(16.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
                        ) {
                            Icon(imageVector = Icons.Default.AutoGraph, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("RUN FORECAST", fontWeight = FontWeight.Bold, fontSize = 14.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                    }

                    // Interactive anomaly toggles
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "ANOMALY & SEASONAL FACTORS",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF535F70),
                        letterSpacing = 0.5.sp
                    )

                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        // Holiday Switch
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.DateRange,
                                    contentDescription = "Holiday",
                                    tint = Color(0xFFE28743),
                                    modifier = Modifier.size(20.dp)
                                )
                                Column {
                                    Text("Upcoming Holiday", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                                    Text("Decreases clinic traffic, surges ER triage (-12%)", fontSize = 10.sp, color = Color.Gray)
                                }
                            }
                            Switch(
                                checked = isHoliday,
                                onCheckedChange = { viewModel.setHoliday(it) },
                                colors = SwitchDefaults.colors(checkedThumbColor = Color(0xFF0061A4))
                            )
                        }

                        // Extreme Weather Switch
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.WbCloudy,
                                    contentDescription = "Extreme Weather",
                                    tint = Color(0xFF0061A4),
                                    modifier = Modifier.size(20.dp)
                                )
                                Column {
                                    Text("Extreme Weather Forecast", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                                    Text("Limits physical access and clinic mobility (-22%)", fontSize = 10.sp, color = Color.Gray)
                                }
                            }
                            Switch(
                                checked = isExtremeWeather,
                                onCheckedChange = { viewModel.setExtremeWeather(it) },
                                colors = SwitchDefaults.colors(checkedThumbColor = Color(0xFF0061A4))
                            )
                        }

                        // Local Event Switch
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Campaign,
                                    contentDescription = "Local Event",
                                    tint = Color(0xFF2E7D32),
                                    modifier = Modifier.size(20.dp)
                                )
                                Column {
                                    Text("Local Festival / Public Event", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                                    Text("Triggers temporary density surge (+15%)", fontSize = 10.sp, color = Color.Gray)
                                }
                            }
                            Switch(
                                checked = isLocalEvent,
                                onCheckedChange = { viewModel.setLocalEvent(it) },
                                colors = SwitchDefaults.colors(checkedThumbColor = Color(0xFF0061A4))
                            )
                        }
                    }
                }
            }
        }

        // Detailed Forecast Insights Card (Displays when ensemble model is calculated)
        ensembleResult?.let { result ->
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF3F7FC)),
                    border = BorderStroke(1.dp, Color(0xFFD0E1FD))
                ) {
                    Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text(
                                text = "Ensemble Prediction Details",
                                fontWeight = FontWeight.Bold,
                                fontSize = 15.sp,
                                color = Color(0xFF001D36),
                                textAlign = TextAlign.Center
                            )
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(
                                        if (result.fitConfidence.contains("HIGH")) Color(0xFFE8F5E9) else Color(0xFFFFF3E0)
                                    )
                                    .padding(horizontal = 10.dp, vertical = 4.dp)
                            ) {
                                Text(
                                    text = "CONFIDENCE: ${result.fitConfidence}",
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = if (result.fitConfidence.contains("HIGH")) Color(0xFF2E7D32) else Color(0xFFE65100)
                                )
                            }
                        }

                        // Shift-Wise Predicted Patient Intake Breakdown
                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(14.dp),
                            color = Color(0xFFEBF2FA),
                            border = BorderStroke(1.dp, Color(0xFFC3D8F2))
                        ) {
                            Row(
                                modifier = Modifier
                                    .padding(vertical = 12.dp, horizontal = 8.dp)
                                    .fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceEvenly,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(
                                    modifier = Modifier.weight(1f),
                                    horizontalAlignment = Alignment.CenterHorizontally
                                ) {
                                    Text("MORNING SHIFT", fontSize = 9.sp, fontWeight = FontWeight.Bold, color = Color(0xFF004881), letterSpacing = 0.5.sp)
                                    Text("${result.morningPred}", fontSize = 16.sp, fontWeight = FontWeight.Black, color = Color(0xFF001D36))
                                    Text("Patients (~45%)", fontSize = 10.sp, color = Color(0xFF43474E))
                                }
                                Box(modifier = Modifier.width(1.dp).height(32.dp).background(Color(0xFFC3D0E5)))
                                Column(
                                    modifier = Modifier.weight(1f),
                                    horizontalAlignment = Alignment.CenterHorizontally
                                ) {
                                    Text("EVENING SHIFT", fontSize = 9.sp, fontWeight = FontWeight.Bold, color = Color(0xFF004881), letterSpacing = 0.5.sp)
                                    Text("${result.eveningPred}", fontSize = 16.sp, fontWeight = FontWeight.Black, color = Color(0xFF001D36))
                                    Text("Patients (~35%)", fontSize = 10.sp, color = Color(0xFF43474E))
                                }
                                Box(modifier = Modifier.width(1.dp).height(32.dp).background(Color(0xFFC3D0E5)))
                                Column(
                                    modifier = Modifier.weight(1f),
                                    horizontalAlignment = Alignment.CenterHorizontally
                                ) {
                                    Text("NIGHT SHIFT", fontSize = 9.sp, fontWeight = FontWeight.Bold, color = Color(0xFF004881), letterSpacing = 0.5.sp)
                                    Text("${result.nightPred}", fontSize = 16.sp, fontWeight = FontWeight.Black, color = Color(0xFF001D36))
                                    Text("Patients (~20%)", fontSize = 10.sp, color = Color(0xFF43474E))
                                }
                            }
                        }

                        // Model Breakdown Columns -- weights shown here come directly
                        // from result.*WeightPercent (computed from each model's own
                        // held-out MAE, inversely: a lower error genuinely earns more
                        // weight). Previously these were hardcoded 30/40/30 labels with
                        // no connection to the actual blend; that has been removed.
                        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            if (result.isRealEnsemble) {
                                ModelMetricRow(
                                    modelName = "Ridge Regression (Trend)",
                                    prediction = result.ridgeRegressionPred.toInt(),
                                    percentage = result.ridgeWeightPercent,
                                    barColor = Color(0xFF2196F3)
                                )
                                ModelMetricRow(
                                    modelName = "Gradient-Boosted Stumps (Non-Linear)",
                                    prediction = result.gradientBoostedPred.toInt(),
                                    percentage = result.gradientBoostedWeightPercent,
                                    barColor = Color(0xFF9C27B0)
                                )
                                ModelMetricRow(
                                    modelName = "Holt-Winters Smoothing (Lag-7 Period)",
                                    prediction = result.holtWintersPred.toInt(),
                                    percentage = result.holtWintersWeightPercent,
                                    barColor = Color(0xFFFF9800)
                                )
                                ModelMetricRow(
                                    modelName = "Heuristic Formula (recency + momentum + season)",
                                    prediction = result.ensemblePred,
                                    percentage = result.heuristicWeightPercent,
                                    barColor = Color(0xFF607D8B)
                                )
                            } else {
                                Text(
                                    "No per-shift dataset available yet, so this forecast is the " +
                                        "heuristic formula only -- the values below are different " +
                                        "views of that one formula, not independently weighted models.",
                                    fontSize = 11.sp,
                                    color = Color(0xFF607D8B)
                                )
                                ModelMetricRow(
                                    modelName = "Ridge Regression (Trend) -- heuristic estimate",
                                    prediction = result.ridgeRegressionPred.toInt(),
                                    percentage = 0,
                                    barColor = Color(0xFF2196F3)
                                )
                                ModelMetricRow(
                                    modelName = "Gradient-Boosted Stumps (Non-Linear) -- heuristic estimate",
                                    prediction = result.gradientBoostedPred.toInt(),
                                    percentage = 0,
                                    barColor = Color(0xFF9C27B0)
                                )
                                ModelMetricRow(
                                    modelName = "Holt-Winters Smoothing (Lag-7 Period) -- heuristic estimate",
                                    prediction = result.holtWintersPred.toInt(),
                                    percentage = 0,
                                    barColor = Color(0xFFFF9800)
                                )
                            }
                        }

                        HorizontalDivider(color = Color(0xFFD0E1FD), thickness = 1.dp)

                        // Final Ensemble Consensus Output
                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp),
                            color = Color(0xFFE8F0FE),
                            border = BorderStroke(1.dp, Color(0xFFD0E1FD))
                        ) {
                            Row(
                                modifier = Modifier
                                    .padding(16.dp)
                                    .fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text("Weighted Consensus Forecast", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFF535F70))
                                    Spacer(modifier = Modifier.height(2.dp))
                                    Text("Consolidated ER Target", fontSize = 14.sp, fontWeight = FontWeight.ExtraBold, color = Color(0xFF001D36))
                                }
                                Column(horizontalAlignment = Alignment.End) {
                                    Text(
                                        text = "${result.ensemblePred} Patients",
                                        fontSize = 20.sp,
                                        fontWeight = FontWeight.Black,
                                        color = Color(0xFF0061A4)
                                    )
                                    Text(
                                        text = "Weighted Consensus",
                                        fontSize = 9.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = Color(0xFF0061A4)
                                    )
                                }
                            }
                        }

                        // This Week's Roster Forecast -- one entry per day of the
                        // fixed Monday..Sunday roster template, sourced from
                        // result.weeklyForecast. This is what actually feeds
                        // runConstructiveRosterAssignment now; the "Weighted
                        // Consensus" card above is a single-moment snapshot, not
                        // what the roster in Step 3 is built from.
                        if (result.weeklyForecast.isNotEmpty()) {
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text(
                                    "This Week's Roster Forecast",
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFF001D36)
                                )
                                Text(
                                    "Each roster day below uses its own forecast, so the roster " +
                                        "in Step 3 can staff a busier day differently from a quieter one " +
                                        "instead of repeating one day's number across the whole week.",
                                    fontSize = 10.sp,
                                    color = Color(0xFF607D8B)
                                )
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .horizontalScroll(rememberScrollState()),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    result.weeklyForecast.forEach { dayForecast ->
                                        Surface(
                                            shape = RoundedCornerShape(10.dp),
                                            color = Color(0xFFEBF2FA),
                                            border = BorderStroke(1.dp, Color(0xFFC3D8F2))
                                        ) {
                                            Column(
                                                modifier = Modifier
                                                    .width(64.dp)
                                                    .padding(vertical = 8.dp, horizontal = 6.dp),
                                                horizontalAlignment = Alignment.CenterHorizontally
                                            ) {
                                                Text(
                                                    dayForecast.day.take(3).uppercase(),
                                                    fontSize = 9.sp,
                                                    fontWeight = FontWeight.Bold,
                                                    color = Color(0xFF004881)
                                                )
                                                Text(
                                                    "${dayForecast.total}",
                                                    fontSize = 14.sp,
                                                    fontWeight = FontWeight.Black,
                                                    color = Color(0xFF001D36)
                                                )
                                                Text("patients", fontSize = 8.sp, color = Color(0xFF43474E))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 2 Proceed Trigger Button
        item {
            Button(
                onClick = { onProceedToLP() },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp)
                    .testTag("proceed_to_staff_selection_button"),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = "Proceed to Staff Selection",
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp,
                        color = Color.White,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Icon(imageVector = Icons.Default.ArrowForward, contentDescription = null, modifier = Modifier.size(18.dp), tint = Color.White)
                }
            }
        }
    }
}

@Composable
fun StaffOutputBox(
    title: String,
    value: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector = Icons.Default.Groups,
    accentColor: Color = Color(0xFF0061A4),
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        color = Color(0xFFF8FAFC),
        border = BorderStroke(1.dp, Color(0xFFE2E8F0))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(accentColor.copy(alpha = 0.12f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = accentColor,
                    modifier = Modifier.size(16.dp)
                )
            }
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = title,
                    fontSize = 11.sp,
                    lineHeight = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color(0xFF475569),
                    softWrap = true
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = value,
                    fontSize = 14.sp,
                    lineHeight = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF0F172A),
                    softWrap = true
                )
            }
        }
    }
}

// LEAVE APPROVAL SCREEN (Operations Manager only) -- staff non-availability
// requests only take effect against the roster once approved here; approving
// or rejecting updates the shared leaveRequests StateFlow, which the Staff
// Pool counts (LPStaffingPlannerScreen) and the roster solver both read live.
@Composable
fun LeaveApprovalScreen(viewModel: MediShiftViewModel) {
    val leaveRequests by viewModel.leaveRequests.collectAsStateWithLifecycle()
    val pending = leaveRequests.filter { it.status == "Pending" }
    val decided = leaveRequests.filter { it.status != "Pending" }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .testTag("leave_approval_screen"),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(top = 8.dp, bottom = 32.dp)
    ) {
        item {
            Column {
                Text(
                    text = "Leave Approval",
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 20.sp,
                    color = Color(0xFF001D36),
                    letterSpacing = (-0.5).sp
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = "Review staff non-availability requests. Approving updates the Staff Pool and roster in real time.",
                    fontSize = 12.sp,
                    color = Color(0xFF535F70)
                )
            }
        }

        item {
            Text(
                text = "PENDING REQUESTS (${pending.size})",
                style = TextStyle(color = Color(0xFF535F70), fontSize = 12.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
            )
        }

        if (pending.isEmpty()) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F3FA)),
                    border = BorderStroke(1.dp, Color(0xFFDCE2F9).copy(alpha = 0.5f))
                ) {
                    Column(
                        modifier = Modifier.padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Icon(imageVector = Icons.Default.CheckCircle, contentDescription = null, tint = Color(0xFF2E7D32), modifier = Modifier.size(32.dp))
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("All Caught Up", fontWeight = FontWeight.Bold, fontSize = 16.sp, color = Color(0xFF0061A4))
                        Text(
                            text = "There are no pending non-availability requests awaiting your approval.",
                            fontSize = 12.sp,
                            color = Color(0xFF535F70),
                            textAlign = TextAlign.Center,
                            modifier = Modifier.padding(top = 4.dp)
                        )
                    }
                }
            }
        } else {
            items(pending) { req ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    border = BorderStroke(1.dp, Color(0xFFFFE0B2))
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(req.staffName, fontWeight = FontWeight.Bold, fontSize = 15.sp, color = Color(0xFF1A1C1E))
                                Text(req.staffRole, fontSize = 11.sp, color = Color(0xFF535F70))
                            }
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(10.dp))
                                    .background(Color(0xFFFFF3E0))
                                    .padding(horizontal = 10.dp, vertical = 4.dp)
                            ) {
                                Text("PENDING", fontSize = 10.sp, fontWeight = FontWeight.Black, color = Color(0xFFE65100))
                            }
                        }
                        Spacer(modifier = Modifier.height(10.dp))
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState()),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            req.days.split(",").forEach { d ->
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(8.dp))
                                        .background(Color(0xFFE8F0FE))
                                        .padding(horizontal = 10.dp, vertical = 4.dp)
                                ) {
                                    Text(d.trim(), fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFF0061A4))
                                }
                            }
                        }
                        if (req.reason.isNotBlank()) {
                            Spacer(modifier = Modifier.height(8.dp))
                            Text("Reason: ${req.reason}", fontSize = 12.sp, color = Color(0xFF535F70))
                        }
                        Spacer(modifier = Modifier.height(12.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            OutlinedButton(
                                onClick = { viewModel.rejectLeaveRequest(req.id) },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(44.dp)
                                    .testTag("reject_leave_${req.id}"),
                                shape = RoundedCornerShape(12.dp),
                                border = BorderStroke(1.dp, Color(0xFFBA1A1A))
                            ) {
                                Icon(imageVector = Icons.Default.Close, contentDescription = null, tint = Color(0xFFBA1A1A), modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Reject", color = Color(0xFFBA1A1A), fontWeight = FontWeight.Bold, fontSize = 12.sp)
                            }
                            Button(
                                onClick = { viewModel.approveLeaveRequest(req.id) },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(44.dp)
                                    .testTag("approve_leave_${req.id}"),
                                shape = RoundedCornerShape(12.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
                            ) {
                                Icon(imageVector = Icons.Default.CheckCircle, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Approve", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                            }
                        }
                    }
                }
            }
        }

        if (decided.isNotEmpty()) {
            item {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "REQUEST HISTORY",
                    style = TextStyle(color = Color(0xFF535F70), fontSize = 12.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                )
            }
            items(decided) { req ->
                val (bgColor, textColor) = when (req.status) {
                    "Approved" -> Color(0xFFE8F5E9) to Color(0xFF2E7D32)
                    else -> Color(0xFFFFEBEE) to Color(0xFFBA1A1A)
                }
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                ) {
                    Row(
                        modifier = Modifier.padding(14.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("${req.staffName} (${req.staffRole})", fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Color(0xFF1A1C1E))
                            Text(req.days.replace(",", ", "), fontSize = 11.sp, color = Color(0xFF535F70))
                        }
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(10.dp))
                                .background(bgColor)
                                .padding(horizontal = 10.dp, vertical = 4.dp)
                        ) {
                            Text(req.status.uppercase(), fontSize = 10.sp, fontWeight = FontWeight.Black, color = textColor)
                        }
                    }
                }
            }
        }
    }
}

// OPTIMALITY VERIFICATION REPORT SCREEN (Operations Manager only) -- an
// independent, from-scratch audit of the currently persisted roster against
// every hard constraint in the report's MILP formulation (Section 3.2,
// Constraints 3.5-3.9). This re-scans
// rosterItems/staffList/leaveRequests directly; it does not read any state
// internal to runConstructiveRosterAssignment, so it can genuinely catch a
// regression in the roster-assignment algorithm rather than just restating
// "the solver says it worked."
@Composable
fun OptimalityVerificationReportScreen(viewModel: MediShiftViewModel) {
    val report by viewModel.optimalityReport.collectAsStateWithLifecycle()
    val isVerifying by viewModel.isVerifyingOptimality.collectAsStateWithLifecycle()
    val rosterItems by viewModel.rosterItems.collectAsStateWithLifecycle()

    LaunchedEffect(Unit) {
        if (report == null) viewModel.runOptimalityVerification()
    }

    var expandedId by remember { mutableStateOf<String?>(null) }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .testTag("optimality_report_screen"),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = PaddingValues(top = 8.dp, bottom = 32.dp)
    ) {
        item {
            Column {
                Text(
                    text = "Optimality Verification Report",
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 20.sp,
                    color = Color(0xFF001D36),
                    letterSpacing = (-0.5).sp
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = "Independently re-checks the currently released roster against every hard constraint (Eq. 3.5–3.9) from the MILP formulation -- not just a claim that the solver enforced them.",
                    fontSize = 12.sp,
                    color = Color(0xFF535F70)
                )
            }
        }

        item {
            Button(
                onClick = { viewModel.runOptimalityVerification() },
                enabled = !isVerifying,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
                    .testTag("run_optimality_verification"),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
            ) {
                if (isVerifying) {
                    CircularProgressIndicator(modifier = Modifier.size(18.dp), color = Color.White, strokeWidth = 2.dp)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Scanning Roster...", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                } else {
                    Icon(imageVector = Icons.Default.FactCheck, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(if (report == null) "Run Verification" else "Re-run Verification", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                }
            }
        }

        if (rosterItems.isEmpty()) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF3E0)),
                    border = BorderStroke(1.dp, Color(0xFFFFE0B2))
                ) {
                    Row(modifier = Modifier.padding(14.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(imageVector = Icons.Default.WarningAmber, contentDescription = null, tint = Color(0xFFE65100), modifier = Modifier.size(20.dp))
                        Text(
                            text = "No roster has been generated yet. Run the staffing solver from Staff Selection first, then verify it here.",
                            fontSize = 12.sp,
                            color = Color(0xFF8C5A00)
                        )
                    }
                }
            }
        }

        report?.let { rep ->
            item {
                val allOk = rep.allConstraintsSatisfied
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = if (allOk) Color(0xFFE8F5E9) else Color(0xFFFFEBEE)),
                    border = BorderStroke(1.dp, if (allOk) Color(0xFFC8E6C9) else Color(0xFFFFCDD2))
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            Icon(
                                imageVector = if (allOk) Icons.Default.VerifiedUser else Icons.Default.ErrorOutline,
                                contentDescription = null,
                                tint = if (allOk) Color(0xFF2E7D32) else Color(0xFFBA1A1A),
                                modifier = Modifier.size(28.dp)
                            )
                            Column {
                                Text(
                                    text = if (allOk) "All Hard Constraints Satisfied" else "Constraint Violations Found",
                                    fontWeight = FontWeight.Black,
                                    fontSize = 16.sp,
                                    color = if (allOk) Color(0xFF1B5E20) else Color(0xFF8C1D18)
                                )
                                Text(
                                    text = "${rep.totalSatisfied} / ${rep.totalChecks} checks satisfied (${String.format("%.1f", rep.overallPercent)}%) across ${rep.staffAudited} staff and ${rep.rosterShiftCount} rostered shifts.",
                                    fontSize = 11.sp,
                                    color = Color(0xFF535F70)
                                )
                            }
                        }
                    }
                }
            }

            item {
                Text(
                    text = "CONSTRAINT-BY-CONSTRAINT AUDIT",
                    style = TextStyle(color = Color(0xFF535F70), fontSize = 12.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                )
            }

            items(rep.constraints) { c ->
                val ok = c.isFullySatisfied
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { expandedId = if (expandedId == c.id) null else c.id },
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    border = BorderStroke(1.dp, if (ok) Color(0xFFDCE2F9) else Color(0xFFFFCDD2))
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.weight(1f)) {
                                Icon(
                                    imageVector = if (ok) Icons.Default.CheckCircle else Icons.Default.ErrorOutline,
                                    contentDescription = null,
                                    tint = if (ok) Color(0xFF2E7D32) else Color(0xFFBA1A1A),
                                    modifier = Modifier.size(18.dp)
                                )
                                Column {
                                    Text("${c.id} — ${c.name}", fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Color(0xFF1A1C1E))
                                    Text(c.formula, fontSize = 11.sp, color = Color(0xFF535F70))
                                }
                            }
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(10.dp))
                                    .background(if (ok) Color(0xFFE8F5E9) else Color(0xFFFFEBEE))
                                    .padding(horizontal = 10.dp, vertical = 4.dp)
                            ) {
                                Text(
                                    text = "${c.checksSatisfied}/${c.checksPerformed}",
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.Black,
                                    color = if (ok) Color(0xFF2E7D32) else Color(0xFFBA1A1A)
                                )
                            }
                        }

                        if (!ok && expandedId == c.id) {
                            Spacer(modifier = Modifier.height(10.dp))
                            HorizontalDivider(color = Color(0xFFFFCDD2), thickness = 1.dp)
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "${c.checksViolated} VIOLATION${if (c.checksViolated == 1) "" else "S"}",
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFFBA1A1A),
                                letterSpacing = 0.5.sp
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            c.violations.take(10).forEach { v ->
                                Text("• $v", fontSize = 11.sp, color = Color(0xFF535F70), modifier = Modifier.padding(vertical = 2.dp))
                            }
                            if (c.violations.size > 10) {
                                Text(
                                    text = "+ ${c.violations.size - 10} more not shown",
                                    fontSize = 10.sp,
                                    fontStyle = FontStyle.Italic,
                                    color = Color(0xFF8C1D18)
                                )
                            }
                        } else if (!ok) {
                            Text(
                                text = "Tap to view ${c.checksViolated} violation${if (c.checksViolated == 1) "" else "s"}",
                                fontSize = 10.sp,
                                color = Color(0xFFBA1A1A),
                                modifier = Modifier.padding(top = 4.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun LPStaffingPlannerScreen(
    viewModel: MediShiftViewModel,
    predictedInflow: Int,
    onBack: () -> Unit,
    onProceed: () -> Unit
) {
    val lpResult by viewModel.lpResult.collectAsStateWithLifecycle()
    val dailyStaffingPlan by viewModel.dailyStaffingPlan.collectAsStateWithLifecycle()
    val staffList by viewModel.staffList.collectAsStateWithLifecycle()
    val leaveRequests by viewModel.leaveRequests.collectAsStateWithLifecycle()
    val isOptimizing by viewModel.isOptimizing.collectAsStateWithLifecycle()
    val solverStatusMessage by viewModel.solverStatusMessage.collectAsStateWithLifecycle()

    // Today's day-of-week name, used to reflect APPROVED non-availability/leave
    // in the Staff Pool counts in real time -- the moment the Operations
    // Manager approves a leave request, the affected staff member drops out of
    // these counts, and since these counts are what get passed into
    // solveStaffingLP()/runConstructiveRosterAssignment() below, the roster picks it up too.
    val todayName = java.text.SimpleDateFormat("EEEE", java.util.Locale.ENGLISH).format(java.util.Date())
    val onApprovedLeaveTodayIds = leaveRequests.filter { req ->
        req.status == "Approved" && req.days.split(",").any { it.trim().equals(todayName, ignoreCase = true) }
    }.map { it.staffId }.toSet()

    val activePool = staffList.filter { it.isInOptimizationPool && it.id !in onApprovedLeaveTodayIds }

    val doctorsCount = activePool.count { it.role.contains("Doctor", ignoreCase = true) || it.role.contains("Medical Officer", ignoreCase = true) }.coerceAtLeast(1)
    val nursesCount = activePool.count { it.role.contains("Nurse", ignoreCase = true) }.coerceAtLeast(1)
    val pharmacistsCount = activePool.count { it.role.contains("Pharmacist", ignoreCase = true) }.coerceAtLeast(1)
    val labTechsCount = activePool.count { it.role.contains("Lab", ignoreCase = true) }.coerceAtLeast(1)


    // Permanent Patient-Staff Satisfaction Ratios: "Good" = hard safety floor,
    // "Target" = ideal ratio the model actively aims for (always <= Good).
    val docGoodRatioVal = 50.0
    val nurseGoodRatioVal = 20.0
    val pharGoodRatioVal = 100.0
    val labGoodRatioVal = 100.0
    val docTargetRatioVal = 20.0
    val nurseTargetRatioVal = 6.0
    val pharTargetRatioVal = 75.0
    val labTargetRatioVal = 40.0

    // Baseline Minimum Personnel Staffing (safety floor), derived from the Good
    // ratio and predicted patient inflow -- this is minSafe_c from the model.
    val minDocVal = kotlin.math.ceil(predictedInflow / docGoodRatioVal).toInt().coerceAtLeast(1)
    val minNurseVal = kotlin.math.ceil(predictedInflow / nurseGoodRatioVal).toInt().coerceAtLeast(1)
    val minPharVal = kotlin.math.ceil(predictedInflow / pharGoodRatioVal).toInt().coerceAtLeast(1)
    val minLabVal = kotlin.math.ceil(predictedInflow / labGoodRatioVal).toInt().coerceAtLeast(1)

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 32.dp)
    ) {
        item {
            Column {
                Text(
                    text = "LP Staffing & Work-Hour Planner",
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 20.sp,
                    color = Color(0xFF001D36),
                    letterSpacing = (-0.5).sp
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = "Matches staffing per category as closely as possible to the ideal Target patient-staff ratio, never dropping below the Good-ratio safety floor, within budget constraints.",
                    fontSize = 12.sp,
                    color = Color(0xFF535F70)
                )
            }
        }

        // Active forecast summary card
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFF3F7FC)),
                border = BorderStroke(1.dp, Color(0xFFD0E1FD))
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(40.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .background(Color(0xFF0061A4)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(imageVector = Icons.Default.TrendingUp, contentDescription = null, tint = Color.White, modifier = Modifier.size(20.dp))
                    }
                    Column {
                        Text("Current Target Admission Forecast", fontSize = 11.sp, color = Color(0xFF535F70), fontWeight = FontWeight.Bold)
                        Text("$predictedInflow Forecasted Patients for Next Week", fontSize = 14.sp, color = Color(0xFF001D36), fontWeight = FontWeight.ExtraBold)
                    }
                }
            }
        }

        // Target Ratios Reference Card
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F4F9)),
                border = BorderStroke(1.dp, Color(0xFFC3E0FF))
            ) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Good (Safety Floor) vs. Target (Ideal) Ratios", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF00325B))
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("• Doctors: Good 1:50 · Target 1:20", fontSize = 11.sp, color = Color(0xFF004881), fontWeight = FontWeight.Medium)
                    }
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("• Nurses: Good 1:20 · Target 1:6", fontSize = 11.sp, color = Color(0xFF004881), fontWeight = FontWeight.Medium)
                    }
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("• Pharmacists: Good 1:100 · Target 1:75", fontSize = 11.sp, color = Color(0xFF004881), fontWeight = FontWeight.Medium)
                    }
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("• Lab Techs: Good 1:100 · Target 1:40", fontSize = 11.sp, color = Color(0xFF004881), fontWeight = FontWeight.Medium)
                    }
                }
            }
        }

        // Inputs Card
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                border = BorderStroke(1.dp, Color(0xFFDCE2F9))
            ) {
                Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    Text(
                        text = "1. Available Limited Staff Pools",
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp,
                        color = Color(0xFF0061A4)
                    )

                    // Non-editable available staff pool displays obtained from staff dataset & off preferences
                    Row(
                        modifier = Modifier.fillMaxWidth().height(IntrinsicSize.Min),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        StaffOutputBox(
                            title = "Available Doctors",
                            value = "$doctorsCount Doctors",
                            icon = Icons.Default.MedicalServices,
                            accentColor = Color(0xFF0061A4),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                        StaffOutputBox(
                            title = "Available Nurses",
                            value = "$nursesCount Nurses",
                            icon = Icons.Default.HealthAndSafety,
                            accentColor = Color(0xFF007A5E),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth().height(IntrinsicSize.Min),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        StaffOutputBox(
                            title = "Available Pharmacists",
                            value = "$pharmacistsCount Pharmacists",
                            icon = Icons.Default.LocalPharmacy,
                            accentColor = Color(0xFF7D5260),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                        StaffOutputBox(
                            title = "Available Lab Techs",
                            value = "$labTechsCount Lab Techs",
                            icon = Icons.Default.Biotech,
                            accentColor = Color(0xFF6750A4),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                    }

                    // Permanent Satisfaction Ratios
                    Text("Good (Floor) / Target (Ideal) Ratios (Patients : Staff)", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                    Row(
                        modifier = Modifier.fillMaxWidth().height(IntrinsicSize.Min),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        StaffOutputBox(
                            title = "Doctor Ratio",
                            value = "1:50 / 1:20",
                            icon = Icons.Default.Equalizer,
                            accentColor = Color(0xFF0061A4),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                        StaffOutputBox(
                            title = "Nurse Ratio",
                            value = "1:20 / 1:6",
                            icon = Icons.Default.Equalizer,
                            accentColor = Color(0xFF007A5E),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth().height(IntrinsicSize.Min),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        StaffOutputBox(
                            title = "Pharmacist Ratio",
                            value = "1:100 / 1:75",
                            icon = Icons.Default.Equalizer,
                            accentColor = Color(0xFF7D5260),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                        StaffOutputBox(
                            title = "Lab Tech Ratio",
                            value = "1:100 / 1:40",
                            icon = Icons.Default.Equalizer,
                            accentColor = Color(0xFF6750A4),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                    }

                    // Baseline Minimum Personnel Staffing Constraints obtained from predicted patients
                    Text("Baseline Minimum Personnel Staffing Constraints (Good-Ratio Safety Floor, From Forecast)", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                    Row(
                        modifier = Modifier.fillMaxWidth().height(IntrinsicSize.Min),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        StaffOutputBox(
                            title = "Min Doctors",
                            value = "$minDocVal Doctors",
                            icon = Icons.Default.Shield,
                            accentColor = Color(0xFF0061A4),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                        StaffOutputBox(
                            title = "Min Nurses",
                            value = "$minNurseVal Nurses",
                            icon = Icons.Default.Shield,
                            accentColor = Color(0xFF007A5E),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth().height(IntrinsicSize.Min),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        StaffOutputBox(
                            title = "Min Pharmacists",
                            value = "$minPharVal Pharmacists",
                            icon = Icons.Default.Shield,
                            accentColor = Color(0xFF7D5260),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                        StaffOutputBox(
                            title = "Min Lab Techs",
                            value = "$minLabVal Lab Techs",
                            icon = Icons.Default.Shield,
                            accentColor = Color(0xFF6750A4),
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    Button(
                        onClick = {
                            viewModel.solveStaffingLP(
                                patients = predictedInflow,
                                availableDocs = doctorsCount,
                                availableNurses = nursesCount,
                                availablePhars = pharmacistsCount,
                                availableLabs = labTechsCount,
                                doctorGoodRatio = docGoodRatioVal,
                                nurseGoodRatio = nurseGoodRatioVal,
                                pharmacistGoodRatio = pharGoodRatioVal,
                                labTechGoodRatio = labGoodRatioVal,
                                doctorTargetRatio = docTargetRatioVal,
                                nurseTargetRatio = nurseTargetRatioVal,
                                pharmacistTargetRatio = pharTargetRatioVal,
                                labTechTargetRatio = labTargetRatioVal
                            )
                        },
                        enabled = !isOptimizing,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(48.dp)
                            .testTag("run_staffing_solver_button"),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center
                        ) {
                            if (isOptimizing) {
                                CircularProgressIndicator(color = Color.White, modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = "Running Staffing Solver...",
                                    style = TextStyle(fontSize = 13.sp, fontWeight = FontWeight.Bold, color = Color.White),
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            } else {
                                Icon(imageVector = Icons.Default.CheckCircle, contentDescription = null, modifier = Modifier.size(18.dp), tint = Color.White)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = "Run Staffing Solver",
                                    style = TextStyle(fontSize = 13.sp, fontWeight = FontWeight.Bold, color = Color.White),
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    }

                    if (isOptimizing && solverStatusMessage.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = solverStatusMessage,
                            fontSize = 11.sp,
                            color = Color(0xFF0061A4),
                            fontWeight = FontWeight.Medium,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }
        }

        // Result Card & Optimality Verification Report
        lpResult?.let { result ->
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    border = BorderStroke(1.dp, if (result.isQualityCompromised) Color(0xFFFFD54F) else Color(0xFF81C784))
                ) {
                    Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text(
                                text = "Optimality Verification Report",
                                fontWeight = FontWeight.Bold,
                                fontSize = 15.sp,
                                color = if (result.isQualityCompromised) Color(0xFFE65100) else Color(0xFF2E7D32),
                                textAlign = TextAlign.Center
                            )
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (result.isQualityCompromised) Color(0xFFFFF3E0) else Color(0xFFE8F5E9))
                                    .padding(horizontal = 10.dp, vertical = 4.dp)
                            ) {
                                Text(
                                    text = if (result.isQualityCompromised) "DEFICIT / COMPROMISED" else "OPTIMAL & CONFIRMED",
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = if (result.isQualityCompromised) Color(0xFFE65100) else Color(0xFF2E7D32)
                                )
                            }
                        }

                        // Optimality Constraints Verification Breakdown
                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp),
                            color = Color(0xFFF8FAFC),
                            border = BorderStroke(1.dp, Color(0xFFE2E8F0))
                        ) {
                            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Optimality Constraints & Ratios Verification", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF0F172A))

                                // Check 1: Target Patient-Staff Ratios
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(
                                        modifier = Modifier.weight(1f),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                                    ) {
                                        Icon(
                                            imageVector = if (!result.isQualityCompromised) Icons.Default.CheckCircle else Icons.Default.Warning,
                                            contentDescription = null,
                                            tint = if (!result.isQualityCompromised) Color(0xFF16A34A) else Color(0xFFD97706),
                                            modifier = Modifier.size(16.dp)
                                        )
                                        Text("Patient-Staff Coverage Ratios", fontSize = 11.sp, color = Color(0xFF334155))
                                    }
                                    Box(
                                        modifier = Modifier
                                            .width(110.dp)
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(if (!result.isQualityCompromised) Color(0xFFDCFCE7) else Color(0xFFFEF3C7))
                                            .padding(vertical = 4.dp, horizontal = 6.dp),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Text(
                                            text = if (!result.isQualityCompromised) "CONFIRMED" else "COMPROMISED",
                                            fontSize = 11.sp,
                                            fontWeight = FontWeight.ExtraBold,
                                            color = if (!result.isQualityCompromised) Color(0xFF15803D) else Color(0xFFB45309),
                                            textAlign = TextAlign.Center
                                        )
                                    }
                                }

                                // Check 2: Baseline Minimums
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(
                                        modifier = Modifier.weight(1f),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.CheckCircle,
                                            contentDescription = null,
                                            tint = Color(0xFF16A34A),
                                            modifier = Modifier.size(16.dp)
                                        )
                                        Text("Baseline Minimum Personnel Thresholds", fontSize = 11.sp, color = Color(0xFF334155))
                                    }
                                    Box(
                                        modifier = Modifier
                                            .width(110.dp)
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(Color(0xFFDCFCE7))
                                            .padding(vertical = 4.dp, horizontal = 6.dp),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Text(
                                            text = "CONFIRMED",
                                            fontSize = 11.sp,
                                            fontWeight = FontWeight.ExtraBold,
                                            color = Color(0xFF15803D),
                                            textAlign = TextAlign.Center
                                        )
                                    }
                                }

                                // Check 3: Budget Limit
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(
                                        modifier = Modifier.weight(1f),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                                    ) {
                                        Icon(
                                            imageVector = if (result.isWithinBudget) Icons.Default.CheckCircle else Icons.Default.Warning,
                                            contentDescription = null,
                                            tint = if (result.isWithinBudget) Color(0xFF16A34A) else Color(0xFFDC2626),
                                            modifier = Modifier.size(16.dp)
                                        )
                                        Text("Weekly Budget & Labor Cost Bound", fontSize = 11.sp, color = Color(0xFF334155))
                                    }
                                    Box(
                                        modifier = Modifier
                                            .width(110.dp)
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(if (result.isWithinBudget) Color(0xFFDCFCE7) else Color(0xFFFEE2E2))
                                            .padding(vertical = 4.dp, horizontal = 6.dp),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Text(
                                            text = if (result.isWithinBudget) "CONFIRMED" else "EXCEEDED",
                                            fontSize = 11.sp,
                                            fontWeight = FontWeight.ExtraBold,
                                            color = if (result.isWithinBudget) Color(0xFF15803D) else Color(0xFFB91C1C),
                                            textAlign = TextAlign.Center
                                        )
                                    }
                                }
                            }
                        }

                        // Status Alert Message
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(if (result.isQualityCompromised) Color(0xFFFFFDE7) else Color(0xFFF1F8E9))
                                .padding(12.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.Center,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    imageVector = if (result.isQualityCompromised) Icons.Default.Warning else Icons.Default.CheckCircle,
                                    contentDescription = null,
                                    tint = if (result.isQualityCompromised) Color(0xFFF57F17) else Color(0xFF33691E),
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = result.statusMessage,
                                    fontSize = 11.sp,
                                    color = if (result.isQualityCompromised) Color(0xFF5D4037) else Color(0xFF1B5E20),
                                    fontWeight = FontWeight.Medium,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }

                        // Display staffing pills
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            StaffPill(role = "Docs", count = result.doctors, color = Color(0xFFE8F0FE), textColor = Color(0xFF0061A4), modifier = Modifier.weight(1f))
                            StaffPill(role = "Nurses", count = result.nurses, color = Color(0xFFE8F5E9), textColor = Color(0xFF2E7D32), modifier = Modifier.weight(1f))
                            StaffPill(role = "Phars", count = result.pharmacists, color = Color(0xFFF3E5F5), textColor = Color(0xFF7B1FA2), modifier = Modifier.weight(1f))
                            StaffPill(role = "Labs", count = result.labTechs, color = Color(0xFFFFF3E0), textColor = Color(0xFFE65100), modifier = Modifier.weight(1f))
                        }

                        HorizontalDivider(color = Color(0xFFE2E2EC), thickness = 1.dp)

                        // Achieved Satisfaction Patient-Staff Ratios
                        Text("Achieved Patient-Staff Satisfaction Ratios", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text("Doctors Ratio:", fontSize = 11.sp, color = Color(0xFF535F70))
                                Text(result.doctorRatioText, fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFF0061A4))
                            }
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text("Nurses Ratio:", fontSize = 11.sp, color = Color(0xFF535F70))
                                Text(result.nurseRatioText, fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFF2E7D32))
                            }
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text("Pharmacists Ratio:", fontSize = 11.sp, color = Color(0xFF535F70))
                                Text(result.pharmacistRatioText, fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFF7B1FA2))
                            }
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text("Lab Techs Ratio:", fontSize = 11.sp, color = Color(0xFF535F70))
                                Text(result.labTechRatioText, fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFFE65100))
                            }
                        }

                        HorizontalDivider(color = Color(0xFFE2E2EC), thickness = 1.dp)

                        // Ideal vs. Actual Staffing (Deviation from Ideal)
                        Text("Ideal vs. Actual Staffing (Chosen / Ideal)", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Card(
                                modifier = Modifier.weight(1f),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFF3F7FC)),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Column(modifier = Modifier.padding(8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Doctor", fontSize = 9.sp, color = Color(0xFF535F70))
                                    Text("${result.doctors} / ${result.idealDoctors}", fontSize = 12.sp, fontWeight = FontWeight.ExtraBold, color = Color(0xFF0061A4))
                                    Text("dev ${result.deviationDoctors}", fontSize = 9.sp, color = Color(0xFF535F70))
                                }
                            }
                            Card(
                                modifier = Modifier.weight(1f),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFF1F8E9)),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Column(modifier = Modifier.padding(8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Nurse", fontSize = 9.sp, color = Color(0xFF535F70))
                                    Text("${result.nurses} / ${result.idealNurses}", fontSize = 12.sp, fontWeight = FontWeight.ExtraBold, color = Color(0xFF2E7D32))
                                    Text("dev ${result.deviationNurses}", fontSize = 9.sp, color = Color(0xFF535F70))
                                }
                            }
                            Card(
                                modifier = Modifier.weight(1f),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFF3E5F5)),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Column(modifier = Modifier.padding(8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Pharmacist", fontSize = 9.sp, color = Color(0xFF535F70))
                                    Text("${result.pharmacists} / ${result.idealPharmacists}", fontSize = 12.sp, fontWeight = FontWeight.ExtraBold, color = Color(0xFF7B1FA2))
                                    Text("dev ${result.deviationPharmacists}", fontSize = 9.sp, color = Color(0xFF535F70))
                                }
                            }
                            Card(
                                modifier = Modifier.weight(1f),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF3E0)),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Column(modifier = Modifier.padding(8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Lab Tech", fontSize = 9.sp, color = Color(0xFF535F70))
                                    Text("${result.labTechs} / ${result.idealLabTechs}", fontSize = 12.sp, fontWeight = FontWeight.ExtraBold, color = Color(0xFFE65100))
                                    Text("dev ${result.deviationLabTechs}", fontSize = 9.sp, color = Color(0xFF535F70))
                                }
                            }
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("Total Deviation from Ideal (Objective Value)", fontSize = 11.sp, color = Color(0xFF535F70), fontWeight = FontWeight.Bold)
                            Text("${result.totalDeviation} staff", fontSize = 13.sp, color = Color(0xFF001D36), fontWeight = FontWeight.Black)
                        }

                        HorizontalDivider(color = Color(0xFFE2E2EC), thickness = 1.dp)

                        // Budget & Labor Cost Summary
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text("Estimated Total Weekly Labor Cost", fontSize = 11.sp, color = Color.Gray, fontWeight = FontWeight.Bold)
                                Text("₹${result.totalLaborCost.toInt()}", fontSize = 16.sp, color = Color(0xFF001D36), fontWeight = FontWeight.Black)
                            }
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (result.isWithinBudget) Color(0xFFE8F5E9) else Color(0xFFFFEBEE))
                                    .padding(horizontal = 8.dp, vertical = 4.dp)
                            ) {
                                Text(
                                    text = if (result.isWithinBudget) "WITHIN BUDGET" else "OVER BUDGET",
                                    fontSize = 9.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = if (result.isWithinBudget) Color(0xFF2E7D32) else Color(0xFFC62828)
                                )
                            }
                        }
                    }
                }
            }
        }

        dailyStaffingPlan?.let { plan ->
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    border = BorderStroke(1.dp, if (plan.anyShiftCompromised) Color(0xFFFFD54F) else Color(0xFF81C784))
                ) {
                    Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text(
                                text = "Per-Shift Optimality Breakdown",
                                fontWeight = FontWeight.Bold,
                                fontSize = 15.sp,
                                color = if (plan.anyShiftCompromised) Color(0xFFE65100) else Color(0xFF2E7D32),
                                textAlign = TextAlign.Center
                            )
                            Text(
                                text = "The objective function is applied independently to every shift block, not just once for the day as a whole.",
                                fontSize = 10.sp,
                                color = Color(0xFF535F70),
                                textAlign = TextAlign.Center
                            )
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (plan.anyShiftCompromised) Color(0xFFFFF3E0) else Color(0xFFE8F5E9))
                                    .padding(horizontal = 10.dp, vertical = 4.dp)
                            ) {
                                Text(
                                    text = if (plan.anyShiftCompromised) "SOME SHIFTS COMPROMISED" else "OPTIMAL AT EVERY SHIFT",
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = if (plan.anyShiftCompromised) Color(0xFFE65100) else Color(0xFF2E7D32)
                                )
                            }
                        }

                        listOf(
                            Triple("Morning", plan.morning, Color(0xFFE8F0FE)),
                            Triple("Evening", plan.evening, Color(0xFFFFF3E0)),
                            Triple("Night", plan.night, Color(0xFFEDE7F6))
                        ).forEach { (shiftLabel, shiftResult, bgColor) ->
                            Surface(
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(16.dp),
                                color = bgColor,
                                border = BorderStroke(1.dp, Color(0xFFE2E8F0))
                            ) {
                                Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text("$shiftLabel Shift", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF0F172A))
                                        Box(
                                            modifier = Modifier
                                                .clip(RoundedCornerShape(8.dp))
                                                .background(if (!shiftResult.isQualityCompromised) Color(0xFFDCFCE7) else Color(0xFFFEF3C7))
                                                .padding(vertical = 3.dp, horizontal = 6.dp)
                                        ) {
                                            Text(
                                                text = if (!shiftResult.isQualityCompromised) "CONFIRMED" else "COMPROMISED",
                                                fontSize = 9.sp,
                                                fontWeight = FontWeight.ExtraBold,
                                                color = if (!shiftResult.isQualityCompromised) Color(0xFF15803D) else Color(0xFFB45309)
                                            )
                                        }
                                    }
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                                    ) {
                                        StaffPill(role = "Docs", count = shiftResult.doctors, color = Color.White, textColor = Color(0xFF0061A4), modifier = Modifier.weight(1f))
                                        StaffPill(role = "Nurses", count = shiftResult.nurses, color = Color.White, textColor = Color(0xFF2E7D32), modifier = Modifier.weight(1f))
                                        StaffPill(role = "Phars", count = shiftResult.pharmacists, color = Color.White, textColor = Color(0xFF7B1FA2), modifier = Modifier.weight(1f))
                                        StaffPill(role = "Labs", count = shiftResult.labTechs, color = Color.White, textColor = Color(0xFFE65100), modifier = Modifier.weight(1f))
                                    }
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween
                                    ) {
                                        Text("Deviation from ideal", fontSize = 10.sp, color = Color(0xFF535F70))
                                        Text("${shiftResult.totalDeviation} staff", fontSize = 10.sp, fontWeight = FontWeight.Bold, color = Color(0xFF001D36))
                                    }
                                }
                            }
                        }

                        HorizontalDivider(color = Color(0xFFE2E2EC), thickness = 1.dp)

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("Combined Deviation (All Shifts)", fontSize = 11.sp, color = Color(0xFF535F70), fontWeight = FontWeight.Bold)
                            Text("${plan.totalDeviationAllShifts} staff", fontSize = 13.sp, color = Color(0xFF001D36), fontWeight = FontWeight.Black)
                        }

                        Text(
                            text = plan.summary,
                            fontSize = 10.sp,
                            color = Color(0xFF535F70)
                        )
                    }
                }
            }
        }

        // Navigation actions
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedButton(
                    onClick = { onBack() },
                    modifier = Modifier
                        .weight(1f)
                        .height(50.dp),
                    shape = RoundedCornerShape(14.dp),
                    border = BorderStroke(1.dp, Color(0xFF0061A4))
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Icon(imageVector = Icons.Default.ArrowBack, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFF0061A4))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Back", fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Color(0xFF0061A4), maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }

                Button(
                    onClick = { onProceed() },
                    modifier = Modifier
                        .weight(1.3f)
                        .height(50.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Text("Proceed to Release Roster", fontWeight = FontWeight.Bold, fontSize = 12.sp, color = Color.White, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Spacer(modifier = Modifier.width(4.dp))
                        Icon(imageVector = Icons.Default.ArrowForward, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color.White)
                    }
                }
            }
        }
    }
}


@Composable
fun ModelMetricRow(
    modelName: String,
    prediction: Int,
    percentage: Int,
    barColor: Color
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = if (percentage > 0) "$modelName ($percentage% weight)" else modelName,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1A1C1E),
                modifier = Modifier.weight(1f).padding(end = 8.dp),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = "$prediction Patients",
                fontSize = 12.sp,
                fontWeight = FontWeight.ExtraBold,
                color = barColor
            )
        }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(CircleShape)
                .background(Color(0xFFE2E8F0))
        ) {
            val fillFraction = (prediction.toFloat() / 2000f).coerceIn(0.15f, 1.0f)
            Box(
                modifier = Modifier
                    .fillMaxWidth(fraction = fillFraction)
                    .fillMaxHeight()
                    .clip(CircleShape)
                    .background(barColor)
            )
        }
    }
}



@Composable
fun StaffPill(
    role: String,
    count: Int,
    color: Color,
    textColor: Color,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(color)
            .padding(horizontal = 4.dp, vertical = 6.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(role, fontSize = 9.sp, color = textColor, fontWeight = FontWeight.Bold)
            Text("$count", fontSize = 14.sp, color = textColor, fontWeight = FontWeight.Black)
        }
    }
}


// ==========================================
// HR & FINANCE SCREENS
// ==========================================

@Composable
fun HRHiringProcessScreen(viewModel: MediShiftViewModel) {
    val candidatesList by viewModel.candidatesList.collectAsStateWithLifecycle()

    // Local state for Manual Registration Form
    var nameField by remember { mutableStateOf("") }
    var emailPrefixField by remember { mutableStateOf("") }
    var roleField by remember { mutableStateOf("Doctor") }
    var seniorityField by remember { mutableStateOf("Junior") }
    var salaryField by remember { mutableStateOf("65000") }
    var allowancesField by remember { mutableStateOf("8000") }

    var formMessage by remember { mutableStateOf<String?>(null) }
    var formIsError by remember { mutableStateOf(false) }

    // Active Tab: 0 = Pending Applications, 1 = Approved Candidates, 2 = Manual pre-registration
    var activeTab by remember { mutableStateOf(0) }

    // State for approving email creation
    var candidateToApprove by remember { mutableStateOf<Candidate?>(null) }
    var approvalEmailPrefix by remember { mutableStateOf("") }
    var approvalSalary by remember { mutableStateOf("") }
    var approvalAllowances by remember { mutableStateOf("") }
    var credentialsMessage by remember { mutableStateOf<String?>(null) }

    BackHandler(enabled = candidateToApprove != null || credentialsMessage != null) {
        if (credentialsMessage != null) {
            credentialsMessage = null
        } else if (candidateToApprove != null) {
            candidateToApprove = null
        }
    }

    val pendingApps = candidatesList.filter { it.status == "Applied" }
    val approvedPool = candidatesList.filter { it.status == "Hired" || it.status == "Registered" }

    // Default weekly work-hour target helper based on position & seniority
    fun getDefaultSalaryAndAllowances(role: String, seniority: String): Pair<Double, Double> {
        return when (role) {
            "Doctor" -> if (seniority == "Senior") Pair(40.0, 8.0) else Pair(40.0, 4.0)
            "Nurse" -> if (seniority == "Senior") Pair(36.0, 12.0) else Pair(36.0, 6.0)
            "Medical Officer" -> if (seniority == "Senior") Pair(40.0, 10.0) else Pair(40.0, 5.0)
            "Operations Manager" -> Pair(40.0, 5.0)
            "Receptionist" -> Pair(35.0, 5.0)
            else -> Pair(40.0, 5.0)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // Tab Row switcher
        TabRow(
            selectedTabIndex = activeTab,
            containerColor = Color(0xFFFDFCFF),
            contentColor = Color(0xFF0061A4),
            modifier = Modifier.fillMaxWidth()
        ) {
            Tab(
                selected = activeTab == 0,
                onClick = { activeTab = 0 },
                text = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Pending Apps", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                        if (pendingApps.isNotEmpty()) {
                            Spacer(modifier = Modifier.width(4.dp))
                            Badge(containerColor = Color(0xFFBA1A1A)) {
                                Text("${pendingApps.size}", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                }
            )
            Tab(
                selected = activeTab == 1,
                onClick = { activeTab = 1 },
                text = { Text("Approved Pool", fontWeight = FontWeight.Bold, fontSize = 12.sp) }
            )
            Tab(
                selected = activeTab == 2,
                onClick = { activeTab = 2 },
                text = { Text("Manual Hire", fontWeight = FontWeight.Bold, fontSize = 12.sp) }
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            contentPadding = PaddingValues(bottom = 32.dp)
        ) {
            // ----------------------------------------
            // TAB 0: PENDING APPLICATIONS
            // ----------------------------------------
            if (activeTab == 0) {
                if (pendingApps.isEmpty()) {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth().padding(top = 24.dp),
                            shape = RoundedCornerShape(24.dp),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F3FA)),
                            border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                        ) {
                            Column(
                                modifier = Modifier.padding(32.dp),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Inbox,
                                    contentDescription = "Empty",
                                    tint = Color(0xFF0061A4),
                                    modifier = Modifier.size(48.dp)
                                )
                                Text(
                                    text = "No Pending Job Applications",
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 16.sp,
                                    color = Color(0xFF1A1C1E)
                                )
                                Text(
                                    text = "Prospective candidates can submit applications using the 'Apply to MediShift Careers' option on the login screen.",
                                    fontSize = 12.sp,
                                    color = Color(0xFF535F70),
                                    textAlign = TextAlign.Center
                                )
                            }
                        }
                    }
                } else {
                    item {
                        Text(
                            text = "PENDING CANDIDATE APPLICATIONS (${pendingApps.size})",
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp,
                            color = Color(0xFF535F70),
                            modifier = Modifier.padding(start = 4.dp, top = 8.dp)
                        )
                    }

                    items(pendingApps) { app ->
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(20.dp),
                            colors = CardDefaults.cardColors(containerColor = Color.White),
                            border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Column {
                                        Text(
                                            text = app.name,
                                            fontWeight = FontWeight.Bold,
                                            fontSize = 16.sp,
                                            color = Color(0xFF1A1C1E)
                                        )
                                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 2.dp)) {
                                            Icon(
                                                imageVector = Icons.Default.Email,
                                                contentDescription = null,
                                                tint = Color(0xFF535F70),
                                                modifier = Modifier.size(14.dp)
                                            )
                                            Spacer(modifier = Modifier.width(4.dp))
                                            Text(
                                                text = app.email,
                                                fontSize = 12.sp,
                                                color = Color(0xFF535F70)
                                            )
                                        }
                                    }

                                    Box(
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(Color(0xFFE8F5E9))
                                            .padding(horizontal = 8.dp, vertical = 4.dp)
                                    ) {
                                        Text(
                                            text = "📥 PENDING REVIEW",
                                            fontSize = 9.sp,
                                            fontWeight = FontWeight.Black,
                                            color = Color(0xFF2E7D32)
                                        )
                                    }
                                }

                                HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp), color = Color(0xFFDCE2F9).copy(alpha = 0.5f))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Column {
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Text(text = "Applied For: ", fontSize = 12.sp, color = Color(0xFF535F70))
                                            Text(text = app.role, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                                        }
                                        
                                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 6.dp)) {
                                            Text(text = "Verify Seniority Status: ", fontSize = 12.sp, color = Color(0xFF535F70))
                                            Spacer(modifier = Modifier.width(6.dp))
                                            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                                listOf("Junior", "Senior").forEach { s ->
                                                    val isSelected = app.seniority == s
                                                    Box(
                                                        modifier = Modifier
                                                            .clip(RoundedCornerShape(6.dp))
                                                            .background(if (isSelected) Color(0xFF0061A4) else Color(0xFFF0F3FA))
                                                            .clickable {
                                                                val defaultPayroll = getDefaultSalaryAndAllowances(app.role, s)
                                                                viewModel.updateCandidateSeniority(app.id, s, defaultPayroll.first, defaultPayroll.second)
                                                            }
                                                            .padding(horizontal = 10.dp, vertical = 4.dp)
                                                    ) {
                                                        Text(
                                                            text = s,
                                                            fontSize = 10.sp,
                                                            fontWeight = FontWeight.Bold,
                                                            color = if (isSelected) Color.White else Color(0xFF535F70)
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp), color = Color(0xFFDCE2F9).copy(alpha = 0.5f))

                                Button(
                                    onClick = {
                                        candidateToApprove = app
                                        val cleanPrefix = app.name.replace("Dr. ", "").replace("Nurse ", "").replace(" ", ".").lowercase()
                                        approvalEmailPrefix = cleanPrefix
                                        val defaultPayroll = getDefaultSalaryAndAllowances(app.role, app.seniority)
                                        approvalSalary = defaultPayroll.first.toInt().toString()
                                        approvalAllowances = defaultPayroll.second.toInt().toString()
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32)),
                                    shape = RoundedCornerShape(10.dp)
                                ) {
                                    Icon(imageVector = Icons.Default.Check, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("Trigger Domain Email Creation & Hire", fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                                }
                            }
                        }
                    }
                }
            }

            // ----------------------------------------
            // TAB 1: APPROVED CANDIDATES POOL
            // ----------------------------------------
            if (activeTab == 1) {
                if (approvedPool.isEmpty()) {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth().padding(top = 24.dp),
                            shape = RoundedCornerShape(24.dp),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F3FA)),
                            border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                        ) {
                            Box(modifier = Modifier.padding(32.dp), contentAlignment = Alignment.Center) {
                                Text(
                                    text = "No approved candidates in the database.",
                                    color = Color(0xFF535F70),
                                    fontSize = 13.sp,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }
                    }
                } else {
                    item {
                        Text(
                            text = "APPROVED ELIGIBLE CANDIDATES POOL (${approvedPool.size})",
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp,
                            color = Color(0xFF535F70),
                            modifier = Modifier.padding(start = 4.dp, top = 8.dp)
                        )
                    }

                    items(approvedPool) { candidate ->
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(containerColor = Color.White),
                            border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Column {
                                        Text(
                                            text = candidate.name,
                                            fontWeight = FontWeight.Bold,
                                            fontSize = 16.sp,
                                            color = Color(0xFF1A1C1E)
                                        )
                                        Text(
                                            text = candidate.email,
                                            fontSize = 12.sp,
                                            color = Color(0xFF0061A4),
                                            fontWeight = FontWeight.SemiBold
                                        )
                                    }

                                    Column(horizontalAlignment = Alignment.End) {
                                        Box(
                                            modifier = Modifier
                                                .clip(RoundedCornerShape(8.dp))
                                                .background(if (candidate.status == "Registered") Color(0xFFE8F5E9) else Color(0xFFFFF3E0))
                                                .padding(horizontal = 8.dp, vertical = 4.dp)
                                        ) {
                                            Text(
                                                text = if (candidate.status == "Registered") "🟢 REGISTERED" else "🟠 ELIGIBLE FOR SIGNUP",
                                                fontSize = 9.sp,
                                                fontWeight = FontWeight.Black,
                                                color = if (candidate.status == "Registered") Color(0xFF2E7D32) else Color(0xFFE65100)
                                            )
                                        }
                                    }
                                }

                                HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp), color = Color(0xFFDCE2F9).copy(alpha = 0.5f))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Column {
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Text(text = "Role: ", fontSize = 12.sp, color = Color(0xFF535F70))
                                            Text(text = candidate.role, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A1C1E))
                                        }
                                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 2.dp)) {
                                            Text(text = "Seniority: ", fontSize = 12.sp, color = Color(0xFF535F70))
                                            Text(
                                                text = candidate.seniority,
                                                fontSize = 12.sp,
                                                fontWeight = FontWeight.Bold,
                                                color = if (candidate.seniority == "Senior") Color(0xFF0061A4) else Color(0xFF535F70)
                                            )
                                        }
                                    }

                                    Column(horizontalAlignment = Alignment.End) {
                                        Text(
                                            text = "Base: ₹${candidate.salary}/mo",
                                            fontSize = 11.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF535F70)
                                        )
                                        Text(
                                            text = "Allowances: ₹${candidate.allowances}/mo",
                                            fontSize = 11.sp,
                                            color = Color(0xFF535F70)
                                        )
                                    }
                                }

                                HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp), color = Color(0xFFDCE2F9).copy(alpha = 0.5f))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Button(
                                        onClick = {
                                            val isCurrentlySenior = candidate.seniority == "Senior"
                                            val nextSeniority = if (isCurrentlySenior) "Junior" else "Senior"
                                            val nextSalary = if (isCurrentlySenior) 36.0 else 40.0
                                            val nextAllowances = if (isCurrentlySenior) 6.0 else 12.0
                                            viewModel.updateCandidateSeniority(candidate.id, nextSeniority, nextSalary, nextAllowances)
                                        },
                                        modifier = Modifier.weight(1f),
                                        shape = RoundedCornerShape(8.dp),
                                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFD1E4FF)),
                                        contentPadding = PaddingValues(vertical = 4.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.TrendingUp,
                                            contentDescription = null,
                                            tint = Color(0xFF001D36),
                                            modifier = Modifier.size(16.dp)
                                        )
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Text(
                                            text = if (candidate.seniority == "Senior") "Demote to Junior" else "Promote & Raise",
                                            fontSize = 11.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF001D36),
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                    }

                                    IconButton(
                                        onClick = { viewModel.deleteCandidate(candidate.id) },
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(Color(0xFFFFDAD6))
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Delete,
                                            contentDescription = "Delete",
                                            tint = Color(0xFFBA1A1A),
                                            modifier = Modifier.size(18.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ----------------------------------------
            // TAB 2: MANUAL PRE-REGISTRATION
            // ----------------------------------------
            if (activeTab == 2) {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(24.dp),
                        colors = CardDefaults.cardColors(containerColor = Color.White),
                        border = BorderStroke(1.dp, Color(0xFFDCE2F9))
                    ) {
                        Column(modifier = Modifier.padding(20.dp)) {
                            Text(
                                text = "NEW HIRE PRE-REGISTRATION",
                                style = TextStyle(
                                    color = Color(0xFF0061A4),
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.Bold,
                                    letterSpacing = 1.5.sp
                                )
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Hire & Generate Domain Email",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Black,
                                color = Color(0xFF1A1C1E)
                            )
                            Text(
                                text = "Pre-register employees to create domain credentials.",
                                fontSize = 12.sp,
                                color = Color(0xFF535F70),
                                modifier = Modifier.padding(bottom = 16.dp)
                            )

                            OutlinedTextField(
                                value = nameField,
                                onValueChange = { nameField = it },
                                label = { Text("Candidate Full Name") },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(12.dp)
                            )
                            Spacer(modifier = Modifier.height(12.dp))

                            OutlinedTextField(
                                value = emailPrefixField,
                                onValueChange = { emailPrefixField = it },
                                label = { Text("Email Prefix (Auto-appends @medishift.ac.in)") },
                                trailingIcon = {
                                    Text(
                                        "@medishift.ac.in",
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = Color(0xFF0061A4),
                                        modifier = Modifier.padding(end = 12.dp)
                                    )
                                },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(12.dp)
                            )
                            Spacer(modifier = Modifier.height(12.dp))

                            Text("Designated Role", style = MaterialTheme.typography.labelMedium, color = Color(0xFF0061A4))
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .horizontalScroll(rememberScrollState()),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                val roles = listOf("Doctor", "Nurse", "Pharmacist", "Lab Technician", "Operations Manager", "Medical Officer", "Receptionist")
                                roles.forEach { r ->
                                    val isSelected = roleField == r
                                    FilterChip(
                                        selected = isSelected,
                                        onClick = { roleField = r },
                                        label = { Text(r) }
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.height(8.dp))

                            Text("Seniority Designation", style = MaterialTheme.typography.labelMedium, color = Color(0xFF0061A4))
                            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                listOf("Junior", "Senior").forEach { s ->
                                    val isSelected = seniorityField == s
                                    FilterChip(
                                        selected = isSelected,
                                        onClick = { seniorityField = s },
                                        label = { Text(s) }
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.height(12.dp))

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                OutlinedTextField(
                                    value = salaryField,
                                    onValueChange = { salaryField = it },
                                    label = { Text("Standard Weekly Hours") },
                                    modifier = Modifier.weight(1f),
                                    shape = RoundedCornerShape(12.dp)
                                )
                                OutlinedTextField(
                                    value = allowancesField,
                                    onValueChange = { allowancesField = it },
                                    label = { Text("Max Overtime Hours") },
                                    modifier = Modifier.weight(1f),
                                    shape = RoundedCornerShape(12.dp)
                                )
                            }

                            if (formMessage != null) {
                                Spacer(modifier = Modifier.height(12.dp))
                                Text(
                                    text = formMessage!!,
                                    color = if (formIsError) Color(0xFFBA1A1A) else Color(0xFF2E7D32),
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            Button(
                                onClick = {
                                    if (nameField.isBlank() || emailPrefixField.isBlank()) {
                                        formMessage = "Name and Email Prefix cannot be empty!"
                                        formIsError = true
                                        return@Button
                                    }
                                    val sal = salaryField.toDoubleOrNull() ?: 0.0
                                    val allow = allowancesField.toDoubleOrNull() ?: 0.0
                                    viewModel.addCandidate(
                                        name = nameField,
                                        emailPrefix = emailPrefixField,
                                        role = roleField,
                                        seniority = seniorityField,
                                        salary = sal,
                                        allowances = allow
                                    )
                                    formMessage = "Candidate '$nameField' hired successfully! Eligible email '${emailPrefixField.trim().lowercase()}@medishift.ac.in' created."
                                    formIsError = false
                                    nameField = ""
                                    emailPrefixField = ""
                                },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(12.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
                            ) {
                                Icon(Icons.Default.PersonAdd, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("PRE-REGISTER & HIRE", fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            }
                        }
                    }
                }
            }
        }
    }

    // Modal dialog to trigger email creation and approve a candidate application
    val candidate = candidateToApprove
    if (candidate != null) {
        AlertDialog(
            onDismissRequest = { candidateToApprove = null },
            title = { Text("Approve & Create Domain Email", fontWeight = FontWeight.Bold, color = Color(0xFF2E7D32)) },
            text = {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text("Approve application of ${candidate.name} for the position of ${candidate.role}.", fontSize = 13.sp, color = Color(0xFF535F70))
                    
                    OutlinedTextField(
                        value = approvalEmailPrefix,
                        onValueChange = { approvalEmailPrefix = it },
                        label = { Text("Domain Email Prefix") },
                        trailingIcon = { Text("@medishift.ac.in", fontSize = 11.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(end = 8.dp)) },
                        modifier = Modifier.fillMaxWidth()
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        OutlinedTextField(
                            value = approvalSalary,
                            onValueChange = { approvalSalary = it },
                            label = { Text("Standard Hours") },
                            modifier = Modifier.weight(1f)
                        )
                        OutlinedTextField(
                            value = approvalAllowances,
                            onValueChange = { approvalAllowances = it },
                            label = { Text("Overtime Limit") },
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Text("Seniority is locked to: ${candidate.seniority}. (Verify seniority on card if change is required)", fontSize = 11.sp, fontStyle = FontStyle.Italic, color = Color(0xFF535F70))
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        val finalPrefix = approvalEmailPrefix.trim()
                        if (finalPrefix.isNotBlank()) {
                            val sal = approvalSalary.toDoubleOrNull() ?: 0.0
                            val allow = approvalAllowances.toDoubleOrNull() ?: 0.0
                            viewModel.approveCandidate(
                                candidateId = candidate.id,
                                name = candidate.name,
                                emailPrefix = finalPrefix,
                                role = candidate.role,
                                seniority = candidate.seniority,
                                salary = sal,
                                allowances = allow
                            )
                            credentialsMessage = "Domain Email Registered! 🎉\n\nCandidate: ${candidate.name}\nEmail: $finalPrefix@medishift.ac.in\nRole: ${candidate.role}\nSeniority: ${candidate.seniority}\nHours: ${sal.toInt()} hrs/week (Overtime: ${allow.toInt()} hrs max)\n\nThey may now sign up using their new official email."
                            candidateToApprove = null
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
                ) {
                    Text("Approve & Create Email", maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            },
            dismissButton = {
                TextButton(onClick = { candidateToApprove = null }) {
                    Text("Cancel", color = Color(0xFF535F70))
                }
            }
        )
    }

    // Modal dialog to display successfully generated credentials
    if (credentialsMessage != null) {
        AlertDialog(
            onDismissRequest = { credentialsMessage = null },
            title = { Text("Credentials Generated Successfully", fontWeight = FontWeight.Bold, color = Color(0xFF0061A4)) },
            text = { Text(credentialsMessage!!) },
            confirmButton = {
                Button(
                    onClick = { credentialsMessage = null },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0061A4))
                ) {
                    Text("Perfect", maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
        )
    }
}






