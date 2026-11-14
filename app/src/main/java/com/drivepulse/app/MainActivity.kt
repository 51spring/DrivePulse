package com.drivepulse.app

import android.Manifest
import android.car.Car
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.drivepulse.app.drive.DemoTrip
import com.drivepulse.app.drive.DriveRecorderService
import com.drivepulse.app.drive.DriveStatus
import com.drivepulse.app.drive.DriveStatusBus
import com.drivepulse.app.drive.PersonalBaseline
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var recordState: TextView
    private lateinit var speedText: TextView
    private lateinit var accelText: TextView
    private lateinit var signalsText: TextView
    private lateinit var eventText: TextView
    private lateinit var messagesText: TextView
    private lateinit var alertBanner: TextView
    private lateinit var tripReportText: TextView
    private lateinit var driftOkButton: Button
    private var answered: PersonalBaseline.DriftAck? = null

    private val required = arrayOf(
        Car.PERMISSION_SPEED,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION,
    )
    private val optional = arrayOf(Manifest.permission.POST_NOTIFICATIONS)

    private var pendingDemo: DemoTrip.Kind? = null

    private val permissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            if (hasRequired()) {
                DriveRecorderService.start(this, pendingDemo)
            } else {
                messagesText.text = "차속·위치 권한이 있어야 기록할 수 있어요."
            }
        }

    private val statusListener: (DriveStatus) -> Unit = { render(it) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        recordState = findViewById(R.id.recordState)
        speedText = findViewById(R.id.speedText)
        accelText = findViewById(R.id.accelText)
        signalsText = findViewById(R.id.signalsText)
        eventText = findViewById(R.id.eventText)
        messagesText = findViewById(R.id.messagesText)
        alertBanner = findViewById(R.id.alertBanner)
        tripReportText = findViewById(R.id.tripReportText)
        driftOkButton = findViewById(R.id.driftOkButton)
        driftOkButton.setOnClickListener {
            val s = DriveStatusBus.latest
            val ack = s.driftAck ?: return@setOnClickListener
            val path = s.historyFile ?: return@setOnClickListener
            PersonalBaseline(File(path)).acknowledge(ack)
            answered = ack
            render(s)
        }

        findViewById<Button>(R.id.startButton).setOnClickListener { startRecording(null) }
        findViewById<Button>(R.id.demoCommuteButton).setOnClickListener { startRecording(DemoTrip.Kind.COMMUTE) }
        findViewById<Button>(R.id.demoDriftButton).setOnClickListener { startRecording(DemoTrip.Kind.DRIFT) }
        findViewById<Button>(R.id.demoHazardButton).setOnClickListener { startRecording(DemoTrip.Kind.HAZARD) }
        findViewById<Button>(R.id.stopButton).setOnClickListener {
            DriveRecorderService.stop(this)
        }
    }

    override fun onStart() {
        super.onStart()
        DriveStatusBus.observe(statusListener)
    }

    override fun onStop() {
        DriveStatusBus.remove(statusListener)
        super.onStop()
    }

    private fun startRecording(demo: DemoTrip.Kind?) {
        if (DriveStatusBus.latest.recording) {
            messagesText.text = "이미 기록 중이에요. 먼저 정지해 주세요."
            return
        }
        pendingDemo = demo
        if (hasRequired()) DriveRecorderService.start(this, demo)
        else permissionRequest.launch(required + optional)
    }

    private fun hasRequired() = required.all {
        checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED
    }

    private fun render(s: DriveStatus) {
        recordState.text = if (s.recording) "● 기록 중" + (s.clock?.let { "  $it" } ?: "") else "대기 중"
        recordState.setTextColor(if (s.recording) Color.rgb(0xD0, 0x30, 0x30) else Color.GRAY)

        alertBanner.text = s.alert.orEmpty()
        alertBanner.visibility = if (s.alert != null) android.view.View.VISIBLE else android.view.View.GONE

        speedText.text = s.speedKmh?.let { "%.1f km/h".format(it) } ?: "-- km/h"
        val accel = s.accelMps2
        accelText.text = accel?.let { "%+.2f m/s²".format(it) } ?: "-- m/s²"
        accelText.setTextColor(if (accel != null && accel <= -3f) Color.rgb(0xD0, 0x30, 0x30) else Color.DKGRAY)

        signalsText.text = buildString {
            appendLine("차속 수신율: ${s.speedRateHz?.let { "%.1f Hz".format(it) } ?: "-"}   샘플: ${s.sampleCount}   주행: ${"%.2f".format(s.distanceKm)} km")
            appendLine("브레이크: ${s.brakePressed?.let { if (it) "밟음" else "뗌" } ?: "-"}   기어: ${s.gear ?: "-"}   조향각: ${s.steeringDeg?.let { "%.1f°".format(it) } ?: "-"}")
            appendLine("위치: ${if (s.lat != null) "%.5f, %.5f".format(s.lat, s.lon) else "-"}   " +
                "도로 구간: ${s.segmentName?.ifEmpty { null } ?: "-"} (${s.segmentId ?: s.geohash ?: "-"})")
            append("서버: ${when (s.serverOk) { true -> "연결됨"; false -> "연결 안 됨"; null -> "-" }}   " +
                "업로드한 구간 통과: ${s.uploadedTraversals}건")
        }

        eventText.text = "급정거 ${s.hardBrakeCount}회" + (s.lastEvent?.let { "\n최근: $it" } ?: "")

        tripReportText.visibility = if (s.tripReport != null) android.view.View.VISIBLE else android.view.View.GONE
        val acked = s.driftAck != null && s.driftAck == answered
        tripReportText.text = s.tripReport?.let {
            "운행 리포트 (평소의 나와 비교)\n$it" + if (acked) "\n→ 알겠어요. 오늘 이후 운전을 새 기준으로 다시 볼게요." else ""
        }.orEmpty()
        driftOkButton.visibility = if (s.driftAck != null && !acked) android.view.View.VISIBLE else android.view.View.GONE
        tripReportText.setBackgroundColor(when (s.tripLevel) {
            "ALERT" -> Color.rgb(0xFD, 0xE2, 0xE1)
            "WATCH" -> Color.rgb(0xFF, 0xF4, 0xD6)
            else -> Color.rgb(0xEE, 0xF2, 0xF7)
        })

        messagesText.text = buildString {
            s.logDir?.let { appendLine("로그: $it") }
            s.messages.forEach { appendLine(it) }
        }
    }
}
