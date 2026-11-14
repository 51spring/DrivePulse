package com.drivepulse.app.drive

import android.content.Context
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Road segments of one area (exported from the SUMO/OSM network by
 * analysis/export_app_assets.py), so the car, the simulator and the server use the same
 * segment IDs.
 *
 * Map matching: nearest segment within [MAX_DIST_M] whose direction agrees with the car's
 * heading; the previous segment gets a small bonus so the match does not flicker at
 * junctions. Coordinates are projected to a local metric plane around the map centre.
 */
class SegmentMap private constructor(val area: String, val segments: List<Segment>) {

    class Segment(val id: String, val name: String, val type: String, val xs: DoubleArray, val ys: DoubleArray)

    data class Match(val segment: Segment, val distanceM: Double)

    private val lat0 = segments.flatMap { s -> s.ys.indices.map { s.ys[it] } }.average()
    private val lon0 = segments.flatMap { s -> s.xs.indices.map { s.xs[it] } }.average()
    private val mPerDegLat = 111_320.0
    private val mPerDegLon = 111_320.0 * cos(Math.toRadians(lat0))
    private val grid = HashMap<Long, MutableList<Segment>>()

    init {
        for (s in segments) {
            val cells = HashSet<Long>()
            for (i in 0 until s.xs.size - 1) {
                val (x1, y1) = local(s.ys[i], s.xs[i])
                val (x2, y2) = local(s.ys[i + 1], s.xs[i + 1])
                val steps = (hypot(x2 - x1, y2 - y1) / (CELL_M / 2)).toInt() + 1
                for (k in 0..steps) {
                    val f = k.toDouble() / steps
                    cells += cell(x1 + f * (x2 - x1), y1 + f * (y2 - y1))
                }
            }
            cells.forEach { grid.getOrPut(it) { mutableListOf() } += s }
        }
    }

    val byId: Map<String, Segment> = segments.associateBy { it.id }

    private fun local(lat: Double, lon: Double) = Pair((lon - lon0) * mPerDegLon, (lat - lat0) * mPerDegLat)

    private fun cell(x: Double, y: Double): Long =
        (Math.floorDiv(x.toLong(), CELL_M.toLong()) shl 32) xor (Math.floorDiv(y.toLong(), CELL_M.toLong()) and 0xffffffffL)

    private fun nearby(x: Double, y: Double): Set<Segment> {
        val out = HashSet<Segment>()
        val cx = Math.floorDiv(x.toLong(), CELL_M.toLong())
        val cy = Math.floorDiv(y.toLong(), CELL_M.toLong())
        for (dx in -1L..1L) for (dy in -1L..1L) {
            grid[((cx + dx) shl 32) xor ((cy + dy) and 0xffffffffL)]?.let { out += it }
        }
        return out
    }

    private class Projection(val distance: Double, val bearing: Double, val pastEnd: Boolean)

    /** Distance (m) to the polyline, its bearing there (deg), and whether the point lies beyond its end. */
    private fun project(s: Segment, x: Double, y: Double): Projection {
        var best = Double.MAX_VALUE
        var bearing = 0.0
        var pastEnd = false
        val last = s.xs.size - 2
        for (i in 0 until s.xs.size - 1) {
            val (x1, y1) = local(s.ys[i], s.xs[i])
            val (x2, y2) = local(s.ys[i + 1], s.xs[i + 1])
            val dx = x2 - x1
            val dy = y2 - y1
            val len2 = dx * dx + dy * dy
            val raw = if (len2 == 0.0) 0.0 else ((x - x1) * dx + (y - y1) * dy) / len2
            val f = raw.coerceIn(0.0, 1.0)
            val d = hypot(x - (x1 + f * dx), y - (y1 + f * dy))
            if (d < best) {
                best = d
                bearing = (Math.toDegrees(atan2(dx, dy)) + 360) % 360
                pastEnd = i == last && raw > 1.0
            }
        }
        return Projection(best, bearing, pastEnd)
    }

    fun match(lat: Double, lon: Double, headingDeg: Double?, previousId: String?): Match? {
        val (x, y) = local(lat, lon)
        var best: Match? = null
        var bestScore = Double.MAX_VALUE
        for (s in nearby(x, y)) {
            val p = project(s, x, y)
            val d = p.distance
            if (d > MAX_DIST_M) continue
            var score = d
            if (headingDeg != null) {
                val diff = angleDiff(p.bearing, headingDeg)
                if (diff > 100) continue           // opposite direction of a two-way road
                score += diff / 90.0 * 15.0
            }
            // Stickiness against flicker, but not once the car has driven past the segment's end.
            if (s.id == previousId && !p.pastEnd) score -= 8.0
            if (s.id == previousId && p.pastEnd) score += 5.0
            if (score < bestScore) {
                bestScore = score
                best = Match(s, d)
            }
        }
        return best
    }

    /** Segments within [rangeM] ahead of the car (inside ±[coneDeg] of its heading). */
    fun ahead(lat: Double, lon: Double, headingDeg: Double, rangeM: Double = 600.0, coneDeg: Double = 50.0): Set<String> {
        val (x, y) = local(lat, lon)
        val out = HashSet<String>()
        for (s in segments) {
            for (i in s.xs.indices) {
                val (px, py) = local(s.ys[i], s.xs[i])
                val d = hypot(px - x, py - y)
                if (d > rangeM) continue
                val toPoint = (Math.toDegrees(atan2(px - x, py - y)) + 360) % 360
                if (d < 30 || angleDiff(toPoint, headingDeg) <= coneDeg) {
                    out += s.id
                    break
                }
            }
        }
        return out
    }

    companion object {
        private const val CELL_M = 100.0
        private const val MAX_DIST_M = 30.0

        fun angleDiff(a: Double, b: Double): Double {
            val d = abs(a - b) % 360
            return if (d > 180) 360 - d else d
        }

        /** Loads assets/segments_<area>.csv; null if the asset is missing. */
        fun load(context: Context, area: String = "gangnam"): SegmentMap? {
            val name = "segments_$area.csv"
            if (context.assets.list("")?.contains(name) != true) return null
            val segs = context.assets.open(name).bufferedReader().useLines { lines ->
                lines.drop(1).mapNotNull { parseCsvLine(it) }.mapNotNull { f ->
                    if (f.size < 6) return@mapNotNull null
                    val pts = f[5].split(';').map { p -> p.trim().split(' ').map { it.toDouble() } }
                    Segment(f[0], f[1], f[2], pts.map { it[0] }.toDoubleArray(), pts.map { it[1] }.toDoubleArray())
                }.toList()
            }
            return SegmentMap(area, segs)
        }

        /** Minimal CSV split with double-quote support (names never contain quotes). */
        internal fun parseCsvLine(line: String): List<String>? {
            if (line.isBlank()) return null
            val out = ArrayList<String>()
            val sb = StringBuilder()
            var quoted = false
            for (c in line) {
                when {
                    c == '"' -> quoted = !quoted
                    c == ',' && !quoted -> { out += sb.toString(); sb.clear() }
                    else -> sb.append(c)
                }
            }
            out += sb.toString()
            return out
        }

        /** Bearing (deg) from point 1 to point 2. */
        fun bearing(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
            val dy = (lat2 - lat1) * 111_320.0
            val dx = (lon2 - lon1) * 111_320.0 * cos(Math.toRadians(lat1))
            return (Math.toDegrees(atan2(dx, dy)) + 360) % 360
        }

        fun distanceM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
            val dy = (lat2 - lat1) * 111_320.0
            val dx = (lon2 - lon1) * 111_320.0 * cos(Math.toRadians(lat1))
            return sqrt(dx * dx + dy * dy)
        }
    }
}
