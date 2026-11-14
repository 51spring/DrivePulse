package com.drivepulse.app.drive

import android.os.Handler
import android.os.Looper
import java.util.concurrent.CopyOnWriteArraySet

/** Snapshot of the recorder, published to the UI. */
data class DriveStatus(
    val recording: Boolean = false,
    val speedKmh: Float? = null,
    val accelMps2: Float? = null,
    val speedRateHz: Float? = null,
    val brakePressed: Boolean? = null,
    val gear: String? = null,
    val steeringDeg: Float? = null,
    val lat: Double? = null,
    val lon: Double? = null,
    val geohash: String? = null,
    val segmentId: String? = null,
    val segmentName: String? = null,
    /** Wall clock used for data (simulated 07:10 start in demo mode). */
    val clock: String? = null,
    val sampleCount: Long = 0,
    val distanceKm: Double = 0.0,
    val hardBrakeCount: Int = 0,
    val lastEvent: String? = null,
    val uploadedTraversals: Int = 0,
    /** null = not contacted yet. */
    val serverOk: Boolean? = null,
    /** Hard-braking spike on a segment ahead, from the server. */
    val alert: String? = null,
    /** End-of-trip comparison with the driver's own commute history. */
    val tripReport: String? = null,
    /** LEARNING / NORMAL / WATCH / ALERT */
    val tripLevel: String? = null,
    /** Long-term change the driver can answer "괜찮아요" to, and the history file it belongs to. */
    val driftAck: PersonalBaseline.DriftAck? = null,
    val historyFile: String? = null,
    val logDir: String? = null,
    val messages: List<String> = emptyList(),
)

/** Tiny in-process bus between the recorder service and the activity. */
object DriveStatusBus {
    private val main = Handler(Looper.getMainLooper())
    private val listeners = CopyOnWriteArraySet<(DriveStatus) -> Unit>()

    @Volatile
    var latest: DriveStatus = DriveStatus()
        private set

    fun publish(status: DriveStatus) {
        latest = status
        main.post { listeners.forEach { it(status) } }
    }

    fun observe(listener: (DriveStatus) -> Unit) {
        listeners += listener
        listener(latest)
    }

    fun remove(listener: (DriveStatus) -> Unit) {
        listeners -= listener
    }
}
