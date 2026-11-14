package com.drivepulse.app.drive

import java.time.Instant
import java.time.ZoneId

/**
 * Turns the matched-segment stream into traversals: one row per segment passed, with the
 * number of hard brakes that started on it. Same unit the server and analysis use.
 *
 * A hard brake is credited to the segment the car was on when the braking *started*
 * (braking for a signal belongs to the road leading into it).
 */
class TraversalTracker(private val tripId: String, private val zone: ZoneId = ZoneId.of("Asia/Seoul")) {

    data class Traversal(
        val tripId: String,
        val segmentId: String,
        val date: String,
        val hour: Int,
        val minute: Int,
        var hardBrakes: Int,
        val enterNanos: Long,
    )

    private val open = ArrayDeque<Traversal>()   // current + a few recent ones
    private val finished = ArrayList<Traversal>()
    private val all = ArrayList<Traversal>()

    val current: Traversal? get() = open.lastOrNull()

    /** Every traversal of the trip so far (for the end-of-trip summary). */
    fun allTraversals(): List<Traversal> = all.toList()

    /** Car is on [segmentId] at [tNanos] (wall time [wall]). */
    fun onSegment(segmentId: String, tNanos: Long, wall: Instant) {
        if (open.lastOrNull()?.segmentId == segmentId) return
        val t = wall.atZone(zone)
        val trav = Traversal(tripId, segmentId, t.toLocalDate().toString(), t.hour, t.minute, 0, tNanos)
        open.addLast(trav)
        all += trav
        // keep the last few open so a late-detected brake can still be credited, then finish
        while (open.size > 3) finished += open.removeFirst()
    }

    /** A hard brake that started at [startNanos]; returns the segment it was credited to. */
    fun onHardBrake(startNanos: Long): String? {
        val trav = open.lastOrNull { it.enterNanos <= startNanos } ?: open.firstOrNull() ?: return null
        trav.hardBrakes++
        return trav.segmentId
    }

    /** Finished traversals since the last call. */
    fun drain(): List<Traversal> {
        val out = ArrayList(finished)
        finished.clear()
        return out
    }

    /** Ends the trip: everything still open is finished. */
    fun close(): List<Traversal> {
        finished += open
        open.clear()
        return drain()
    }
}
