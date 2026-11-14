package com.drivepulse.app.drive

import android.content.Context

/**
 * A simulated drive through a real Korean road network, recorded at 10 Hz by
 * analysis/export_app_assets.py (SUMO, OpenStreetMap). Replayed in simulation mode
 * because the Pleos emulator cannot change vehicle speed or follow a route.
 */
class DemoTrip(val area: String, val kind: Kind, val samples: List<Sample>, val startMinuteOfDay: Int) {

    data class Sample(val tSec: Double, val lat: Double, val lon: Double, val speedMps: Float, val headingDeg: Float)

    /**
     * COMMUTE: a normal driver's morning commute on a tired day (personal-baseline demo,
     *   assets/demo_commute_<area>.csv, with past commutes in demo_history_<area>.csv).
     * DRIFT: the same tired commute after two weeks of commutes that got more and more
     *   fatigued (long-term change demo, history in demo_history_drift_<area>.csv).
     * HAZARD: a car crossing Teheran-ro that meets an obstacle (segment warning demo,
     *   assets/demo_trip_<area>.csv). Another car's drive: no commute history.
     */
    enum class Kind(val asset: String, val label: String, val history: String?) {
        COMMUTE("demo_commute", "피곤한 날 출근길", "demo_history"),
        DRIFT("demo_commute", "몇 주째 피곤한 출근길", "demo_history_drift"),
        HAZARD("demo_trip", "전방 구간 경고", null),
    }

    companion object {
        /** Simulated clock: the SUMO trip departs at 07:10 (morning rush). */
        const val START_MINUTE_OF_DAY = 7 * 60 + 10

        fun load(context: Context, kind: Kind, area: String = "gangnam"): DemoTrip? {
            val name = "${kind.asset}_$area.csv"
            if (context.assets.list("")?.contains(name) != true) return null
            val samples = context.assets.open(name).bufferedReader().useLines { lines ->
                lines.drop(1).mapNotNull { line ->
                    val f = line.split(',')
                    if (f.size < 5) null
                    else Sample(f[0].toDouble(), f[1].toDouble(), f[2].toDouble(), f[3].toFloat(), f[4].toFloat())
                }.toList()
            }
            return DemoTrip(area, kind, samples, START_MINUTE_OF_DAY)
        }
    }
}
