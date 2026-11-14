package com.drivepulse.app.drive

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class HardBrakeDetectorTest {

    private val hz = 10
    private val stepNanos = 1_000_000_000L / hz

    /** Runs a speed profile (m/s per sample) through a detector and returns all events. */
    private fun run(profile: List<Float>, detector: HardBrakeDetector = HardBrakeDetector()) =
        profile.mapIndexedNotNull { i, v -> detector.onSample(SpeedSample(i * stepNanos, v)) }

    private fun cruise(speedMps: Float, seconds: Float) = List((seconds * hz).toInt()) { speedMps }

    private fun ramp(from: Float, to: Float, accelMps2: Float): List<Float> {
        val out = mutableListOf<Float>()
        var v = from
        val dv = accelMps2 / hz
        while (if (dv < 0) v + dv > to else v + dv < to) { v += dv; out += v }
        out += to
        return out
    }

    @Test
    fun hardStopFrom60IsOneEvent() {
        val v60 = 60f / 3.6f
        val events = run(cruise(v60, 3f) + ramp(v60, 0f, -6f) + cruise(0f, 2f))
        assertEquals(1, events.size)
        val e = events.single()
        assertTrue("peak ${e.peakDecelMps2}", e.peakDecelMps2 in 5f..7f)
        assertTrue("start ${e.startSpeedMps}", e.startSpeedMps > 15f)
        assertTrue("end ${e.endSpeedMps}", e.endSpeedMps < 1f)
    }

    @Test
    fun gentleStopIsNotAnEvent() {
        val v60 = 60f / 3.6f
        assertEquals(0, run(cruise(v60, 3f) + ramp(v60, 0f, -1.5f) + cruise(0f, 2f)).size)
    }

    @Test
    fun hardSlowdownWithoutStoppingIsDetected() {
        val v80 = 80f / 3.6f
        val v40 = 40f / 3.6f
        val events = run(cruise(v80, 3f) + ramp(v80, v40, -4.5f) + cruise(v40, 3f))
        assertEquals(1, events.size)
        assertTrue(events.single().endSpeedMps in (v40 - 1f)..(v40 + 1f))
    }

    @Test
    fun brakingAtCrawlSpeedIsIgnored() {
        val v8 = 8f / 3.6f
        assertEquals(0, run(cruise(v8, 2f) + ramp(v8, 0f, -5f) + cruise(0f, 2f)).size)
    }

    @Test
    fun twoSeparateHardBrakesAreTwoEvents() {
        val v50 = 50f / 3.6f
        val v20 = 20f / 3.6f
        val profile = cruise(v50, 2f) + ramp(v50, v20, -5f) + cruise(v20, 2f) +
            ramp(v20, v50, 2f) + cruise(v50, 2f) + ramp(v50, 0f, -5f) + cruise(0f, 1f)
        assertEquals(2, run(profile).size)
    }

    @Test
    fun accelerationEstimateTracksConstantSlope() {
        val d = HardBrakeDetector()
        run(cruise(20f, 1f) + ramp(20f, 10f, -2f), d)
        assertNotNull(d.lastAccelMps2)
        assertEquals(-2f, d.lastAccelMps2!!, 0.3f)
    }

    @Test
    fun geohashMatchesKnownValue() {
        // Well-known test vector: 57.64911, 10.40744 -> u4pruydqqvj
        assertEquals("u4pruyd", Geohash.encode(57.64911, 10.40744, 7))
    }
}
