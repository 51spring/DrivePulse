package com.drivepulse.app.drive

import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.Instant

class TraversalTrackerTest {

    private val s = 1_000_000_000L
    private val t0 = Instant.parse("2026-11-06T22:10:00Z") // 07:10 KST next day

    @Test
    fun brakeIsCreditedToSegmentWhereItStarted() {
        val tr = TraversalTracker("trip1")
        tr.onSegment("A", 0, t0)
        tr.onSegment("B", 10 * s, t0.plusSeconds(10))
        // braking started on A (t=8 s) but was detected after entering B
        assertEquals("A", tr.onHardBrake(8 * s))
        assertEquals("B", tr.onHardBrake(12 * s))
        val all = tr.close()
        assertEquals(listOf("A" to 1, "B" to 1), all.map { it.segmentId to it.hardBrakes })
    }

    @Test
    fun usesKoreanDateAndTime() {
        val tr = TraversalTracker("trip1")
        tr.onSegment("A", 0, t0)
        val a = tr.close().single()
        assertEquals("2026-11-07", a.date)
        assertEquals(7, a.hour)
        assertEquals(10, a.minute)
    }

    @Test
    fun repeatedSegmentIsOneTraversalAndOldOnesFinish() {
        val tr = TraversalTracker("trip1")
        listOf("A", "A", "B", "C", "D", "E").forEachIndexed { i, id -> tr.onSegment(id, i * s, t0.plusSeconds(i.toLong())) }
        assertEquals(listOf("A", "B"), tr.drain().map { it.segmentId })   // last 3 stay open
        assertEquals(listOf("C", "D", "E"), tr.close().map { it.segmentId })
    }

    @Test
    fun segmentAnglesWrapAround() {
        assertEquals(20.0, SegmentMap.angleDiff(350.0, 10.0), 1e-9)
        assertEquals(180.0, SegmentMap.angleDiff(0.0, 180.0), 1e-9)
    }
}
