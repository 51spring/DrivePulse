package com.drivepulse.app.drive

import org.junit.Assert.assertEquals
import org.junit.Test

class AlertTextTest {

    private fun alert(observed: Int, expected: Double) =
        ServerClient.Alert("s", "테헤란로", "07:10", observed, expected, observed / expected)

    @Test
    fun showsRatioWhenModerate() {
        assertEquals("평소의 3.0배", alert(12, 4.0).spikeText)
    }

    @Test
    fun showsCountsWhenUsualLevelIsNearZero() {
        assertEquals("급정거 52회, 평소엔 거의 없음", alert(52, 0.02).spikeText)
        assertEquals("급정거 40회, 평소 2회", alert(40, 2.0).spikeText)
    }
}
