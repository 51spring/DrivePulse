package com.drivepulse.app.drive

import kotlin.math.max

/** One vehicle speed reading. [tNanos] is SystemClock.elapsedRealtimeNanos-based. */
data class SpeedSample(val tNanos: Long, val speedMps: Float)

data class HardBrakeEvent(
    val startNanos: Long,
    val endNanos: Long,
    val startSpeedMps: Float,
    val endSpeedMps: Float,
    /** Largest deceleration seen during the event, positive m/s². */
    val peakDecelMps2: Float,
    /** (startSpeed - endSpeed) / duration, positive m/s². */
    val meanDecelMps2: Float,
) {
    val durationSec: Float get() = (endNanos - startNanos) / 1e9f
}

/**
 * Detects hard-braking events from a speed time series.
 *
 * Acceleration is the least-squares slope of speed over the last [windowNanos],
 * which smooths sensor noise. An event starts when deceleration reaches
 * [startDecelMps2] while the car was moving at least [minSpeedMps], and ends
 * when deceleration falls below [endDecelMps2] (hysteresis) or the car stops.
 */
class HardBrakeDetector(
    private val startDecelMps2: Float = 3.0f,          // ≈ 0.3 g
    private val endDecelMps2: Float = 1.5f,
    private val minSpeedMps: Float = 10f / 3.6f,       // 10 km/h
    private val windowNanos: Long = 500_000_000L,      // 0.5 s
    private val minDurationNanos: Long = 300_000_000L, // 0.3 s
    private val maxGapNanos: Long = 2_000_000_000L,    // reset if no data for 2 s
) {
    private class Active(val startNanos: Long, val startSpeedMps: Float, var peakDecel: Float)

    private val window = ArrayDeque<SpeedSample>()
    private var active: Active? = null

    /** False right after an event; re-armed once deceleration drops below the end threshold. */
    private var armed = true

    /** Latest smoothed acceleration (m/s², negative = slowing down), or null before enough data. */
    var lastAccelMps2: Float? = null
        private set

    val isBraking: Boolean get() = active != null

    fun reset() {
        window.clear()
        active = null
        armed = true
        lastAccelMps2 = null
    }

    /** Feeds one sample; returns a finished event when one completes. */
    fun onSample(sample: SpeedSample): HardBrakeEvent? {
        val last = window.lastOrNull()
        if (last != null) {
            if (sample.tNanos <= last.tNanos) return null
            if (sample.tNanos - last.tNanos > maxGapNanos) reset()
        }
        window.addLast(sample)
        while (window.size > 2 && sample.tNanos - window.first().tNanos > windowNanos) {
            window.removeFirst()
        }

        val accel = slope() ?: return null
        lastAccelMps2 = accel
        val decel = -accel

        val current = active
        if (current == null) {
            if (!armed) {
                if (decel < endDecelMps2) armed = true
                return null
            }
            val windowMaxSpeed = window.maxOf { it.speedMps }
            if (decel >= startDecelMps2 && windowMaxSpeed >= minSpeedMps && sample.speedMps >= 0.5f) {
                active = Active(window.first().tNanos, windowMaxSpeed, decel)
            }
            return null
        }

        current.peakDecel = max(current.peakDecel, decel)
        if (decel >= endDecelMps2 && sample.speedMps >= 0.5f) return null

        active = null
        armed = false
        val durationNanos = sample.tNanos - current.startNanos
        if (durationNanos < minDurationNanos) return null
        return HardBrakeEvent(
            startNanos = current.startNanos,
            endNanos = sample.tNanos,
            startSpeedMps = current.startSpeedMps,
            endSpeedMps = sample.speedMps,
            peakDecelMps2 = current.peakDecel,
            meanDecelMps2 = (current.startSpeedMps - sample.speedMps) / (durationNanos / 1e9f),
        )
    }

    /** Least-squares slope of speed vs. time in m/s². */
    private fun slope(): Float? {
        if (window.size < 2) return null
        val t0 = window.first().tNanos
        var sumT = 0.0
        var sumV = 0.0
        for (s in window) {
            sumT += (s.tNanos - t0) / 1e9
            sumV += s.speedMps
        }
        val n = window.size
        val meanT = sumT / n
        val meanV = sumV / n
        var num = 0.0
        var den = 0.0
        for (s in window) {
            val dt = (s.tNanos - t0) / 1e9 - meanT
            num += dt * (s.speedMps - meanV)
            den += dt * dt
        }
        if (den <= 0.0) return null
        return (num / den).toFloat()
    }
}
