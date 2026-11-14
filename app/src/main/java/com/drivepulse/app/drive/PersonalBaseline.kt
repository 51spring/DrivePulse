package com.drivepulse.app.drive

import java.io.File
import java.util.Locale
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * The driver's own commute history, kept on the device, and the comparison of a finished
 * commute with it (same window, same route). Mirrors analysis/personal.py; the settings are
 * checked on a year of commutes in analysis/personal_year.py (app_check).
 *
 *  - One-day check: is today's commute unusual for THIS driver (tired / rushed day)?
 *    Hard brakes vs own rate x today's road expectation, plus moderate braking, accel RMS
 *    and max deceleration vs the last [HISTORY_TRIPS] commutes on this route (Fisher-combined).
 *  - Long-term check: CUSUM of daily deviations vs a reference that moves along (80-20 days ago).
 *
 * Personal data never leaves the car; the server is only asked how brake-prone the
 * traversed road segments are usually and today (road-condition adjustment).
 */
class PersonalBaseline(val file: File) {

    /** Days the driver answered "괜찮아요" to a long-term change: window,route,date per line. */
    private val ackFile = File(file.parentFile, file.nameWithoutExtension + "_ack.csv")

    /** A long-term change the driver can acknowledge (the check then starts over after [date]). */
    data class DriftAck(val window: String, val route: String, val date: String)

    data class TripSummary(
        val date: String,
        val window: String,          // "am" / "pm" commute window, "other" otherwise
        val routeKey: String,        // first>last road segment
        val km: Double,
        val hardBrakes: Int,
        val moderateBrakes: Int,
        val accelRms: Double,
        val maxDecel: Double,
        val eUsual: Double?,         // expected hard brakes on this route at this time, usual days
        val eToday: Double?,         // same, from other cars today
    ) {
        val moderatePerKm get() = moderateBrakes / km.coerceAtLeast(0.1)

        fun toCsv() = listOf(date, window, routeKey, "%.3f".format(Locale.US, km), hardBrakes, moderateBrakes,
            "%.4f".format(Locale.US, accelRms), "%.3f".format(Locale.US, maxDecel),
            eUsual?.let { "%.4f".format(Locale.US, it) } ?: "", eToday?.let { "%.4f".format(Locale.US, it) } ?: "")
            .joinToString(",")

        companion object {
            const val HEADER = "date,window,route,km,hard_brakes,moderate_brakes,accel_rms,max_decel,e_usual,e_today"

            fun fromCsv(line: String): TripSummary? {
                val f = line.split(',')
                if (f.size < 10) return null
                return runCatching {
                    TripSummary(f[0], f[1], f[2], f[3].toDouble(), f[4].toInt(), f[5].toInt(), f[6].toDouble(),
                        f[7].toDouble(), f[8].toDoubleOrNull(), f[9].toDoubleOrNull())
                }.getOrNull()
            }
        }
    }

    enum class Level { LEARNING, NORMAL, WATCH, ALERT }

    /**
     * Likely kind of an unusual day, from which features moved most vs own history:
     * jerky acceleration (accel RMS) more than moderate braking -> rushed; otherwise tired
     * (late reactions: more hard and moderate braking). 84% right on the planted tired /
     * rushed days of the 1-year SUMO evaluation (tired 94%, rushed 76%).
     */
    enum class Cause { TIRED, RUSHED }

    data class Report(
        val level: Level,
        val day: String,
        val drift: String?,
        val observed: Int,
        val expected: Double?,
        val historyTrips: Int,
        val cause: Cause? = null,
        val driftAck: DriftAck? = null,
    )

    fun history(): List<TripSummary> =
        if (!file.exists()) emptyList()
        else file.readLines().drop(1).mapNotNull { TripSummary.fromCsv(it) }

    fun clear() {
        file.delete()
        ackFile.delete()
    }

    /** The driver says the change is fine (new job, calmer habit...): look for changes only after [ack]'s date. */
    fun acknowledge(ack: DriftAck) {
        ackFile.parentFile?.mkdirs()
        ackFile.appendText("${ack.window},${ack.route},${ack.date}\n")
    }

    /** Latest acknowledged date per (window, route). */
    fun acknowledged(): Map<Pair<String, String>, String> =
        if (!ackFile.exists()) emptyMap()
        else ackFile.readLines().mapNotNull { line ->
            val i = line.indexOf(','); val j = line.lastIndexOf(',')
            if (i < 0 || j <= i) null else (line.substring(0, i) to line.substring(i + 1, j)) to line.substring(j + 1)
        }.groupBy({ it.first }, { it.second }).mapValues { (_, v) -> v.max() }

    fun append(t: TripSummary) {
        val isNew = !file.exists()
        file.parentFile?.mkdirs()
        file.appendText((if (isNew) TripSummary.HEADER + "\n" else "") + t.toCsv() + "\n")
    }

    /** Compares [today] with past commutes (earlier dates only). */
    fun evaluate(today: TripSummary, all: List<TripSummary> = history()): Report {
        val label = if (today.window == "am") "출근길" else if (today.window == "pm") "퇴근길" else "이번 운행"
        if (today.window == "other") {
            return Report(Level.NORMAL, "출퇴근 시간대가 아니라 비교하지 않았어요.", null, today.hardBrakes, null, 0)
        }
        val sameWindow = all.filter { it.window == today.window && it.date < today.date }.sortedBy { it.date }
        // Only the same route: borrowing another route's history after a move to a new workplace
        // multiplies false alarms (1-year SUMO evaluation: 2.2 vs 0.4 per 100 commutes).
        val past = sameWindow.filter { it.routeKey == today.routeKey }.takeLast(HISTORY_TRIPS)
        if (past.size < MIN_HISTORY) {
            val msg = if (sameWindow.size >= MIN_HISTORY)
                "새 $label 경로예요. 이 경로의 평소 패턴을 새로 배우는 중이에요 (${past.size}/${MIN_HISTORY}회)."
            else "$label 기록을 모으는 중이에요 (${past.size}/${MIN_HISTORY}회). 평소 패턴을 배우고 있어요."
            return Report(Level.LEARNING, msg, null, today.hardBrakes, null, past.size)
        }

        // expected hard brakes: own rate x today's road expectation (fallback: per km)
        val e = { t: TripSummary -> t.eToday ?: t.eUsual ?: (t.km * DEFAULT_RATE_PER_KM) }
        val f = (past.sumOf { it.hardBrakes } + A_PERSONAL) / (past.sumOf { e(it) } + A_PERSONAL)
        val expected = f * e(today)
        val pHard = hardBrakeUpper(today.hardBrakes, expected, past.map { it.hardBrakes })

        val featureP = listOf(
            zUpper(today.moderatePerKm, past.map { it.moderatePerKm }, POP_VAR_MODERATE),
            zUpper(today.accelRms, past.map { it.accelRms }, POP_VAR_ACCEL_RMS),
            zUpper(today.maxDecel.coerceAtMost(MAX_DECEL_CAP), past.map { it.maxDecel.coerceAtMost(MAX_DECEL_CAP) },
                POP_VAR_MAX_DECEL),
        )
        val pAll = fisher(listOf(pHard) + featureP)
        val level = when {
            pAll < P_ALERT -> Level.ALERT
            pAll < P_WATCH -> Level.WATCH
            else -> Level.NORMAL
        }
        val ratio = (today.hardBrakes + 0.5) / (expected + 0.5)
        val cause = if (level == Level.ALERT || level == Level.WATCH) {
            val zAccel = z(today.accelRms, past.map { it.accelRms }, POP_VAR_ACCEL_RMS)
            val zModerate = z(today.moderatePerKm, past.map { it.moderatePerKm }, POP_VAR_MODERATE)
            if (zAccel > zModerate) Cause.RUSHED else Cause.TIRED
        } else null
        val day = when (level) {
            Level.ALERT -> "오늘 $label 운전이 평소와 많이 달라요. 급정거 ${today.hardBrakes}회 (평소 이 시간 %.1f회). "
                .format(expected) + when (cause) {
                    Cause.RUSHED -> "가속과 감속이 평소보다 거칠었어요. 서두르셨다면 다음엔 조금 여유 있게 출발해 보세요."
                    else -> "급정거와 감속이 평소보다 잦았어요. 피곤하면 반응이 늦어질 수 있으니 잠시 쉬어 가세요."
                }
            Level.WATCH -> "오늘 $label 급정거·급감속이 평소보다 조금 많았어요 (급정거 ${today.hardBrakes}회, 평소 %.1f회). "
                .format(expected) + if (cause == Cause.RUSHED) "가감속이 조금 거칠었어요." else "감속이 조금 잦았어요."
            else -> "오늘 $label 운전은 평소와 비슷했어요 (급정거 ${today.hardBrakes}회, 평소 %.1f회, 비율 %.1f)."
                .format(expected, ratio)
        }
        val since = acknowledged()[today.window to today.routeKey]
        val drift = driftMessage(all + today, today.window, today.routeKey, label, since)
        return Report(level, day, drift, today.hardBrakes, expected, past.size, cause,
            drift?.let { DriftAck(today.window, today.routeKey, today.date) })
    }

    /**
     * CUSUM over the commute days of this window and route; null if no sustained change now.
     * Reference for day i: days i-80 .. i-20 (the first [REF_DAYS] days early on), so a calm
     * first few weeks can't keep the alarm on all year and seasons are absorbed. Each day's
     * deviation is capped at [DAY_Z_CAP] (one bad day is the one-day check's job) and the sum
     * restarts after each alarm; a change is reported while alarms keep coming. Days whose road
     * was unusually gentle (server: today's expectation < [ROAD_SKIP_RATIO] of usual, e.g. snow,
     * everyone driving slowly) are left out: in the reference they make ordinary days look worse.
     */
    fun driftMessage(all: List<TripSummary>, window: String, route: String, label: String, since: String? = null): String? {
        val byDate = all.filter { it.window == window && it.routeKey == route && !gentleRoad(it) &&
            (since == null || it.date > since) }
            .groupBy { it.date }.toSortedMap()
        val dates = byDate.keys.toList()
        val days = byDate.values.toList()
        val n = days.size
        if (n < REF_DAYS + 3) return null
        val e = { t: TripSummary -> t.eToday ?: t.eUsual ?: (t.km * DEFAULT_RATE_PER_KM) }
        var s = 0.0
        var lastZero: String? = null
        var episodeFrom: String? = null
        var lastAlarm = -1_000
        for (i in REF_DAYS until n) {
            val b = minOf(maxOf(REF_DAYS, i - LAG_GAP), n)
            val ref = days.subList(maxOf(0, b - LAG_LEN), b)
            val refTrips = ref.flatten()
            val f = (refTrips.sumOf { it.hardBrakes } + A_PERSONAL) / (refTrips.sumOf { e(it) } + A_PERSONAL)
            val daily = { trips: List<TripSummary>, sel: (TripSummary) -> Double -> trips.map(sel).average() }
            val refDaily = { sel: (TripSummary) -> Double -> ref.map { daily(it, sel) } }
            val trips = days[i]
            val y = trips.sumOf { it.hardBrakes }
            val exp = f * trips.sumOf { e(it) }
            val cap = { v: Double -> v.coerceIn(-Z_CAP, Z_CAP) }
            val zs = listOf(
                (y - exp) / sqrt(exp + 0.25),
                cap(z(daily(trips) { it.moderatePerKm }, refDaily { it.moderatePerKm }, POP_VAR_MODERATE)),
                cap(z(daily(trips) { it.accelRms }, refDaily { it.accelRms }, POP_VAR_ACCEL_RMS)),
                cap(z(daily(trips) { minOf(it.maxDecel, MAX_DECEL_CAP) },
                    refDaily { minOf(it.maxDecel, MAX_DECEL_CAP) }, POP_VAR_MAX_DECEL)),
            )
            val dayZ = (zs.sum() / sqrt(zs.size.toDouble())).coerceIn(-DAY_Z_CAP, DAY_Z_CAP)
            s = max(0.0, s + dayZ - CUSUM_K)
            if (s == 0.0) lastZero = dates[i]
            if (s > CUSUM_H) {
                if (i - lastAlarm > RECENT_DAYS) episodeFrom = lastZero ?: dates[i]
                lastAlarm = i
                s = 0.0
            }
        }
        if (lastAlarm < n - RECENT_DAYS) return null
        return "최근 $label 운전이 꾸준히 달라지고 있어요 ($episodeFrom 무렵부터 급정거·급감속 증가). " +
            "피로가 쌓였거나 운전 습관이 바뀌었는지 살펴보세요."
    }

    companion object {
        const val MIN_HISTORY = 3
        const val HISTORY_TRIPS = 60   // one-day check: the last 60 commutes on this route
        const val A_PERSONAL = 2.0
        const val N0_FEATURE = 3.0
        const val N0_DISP = 10.0
        const val DEFAULT_RATE_PER_KM = 0.15
        // Checked on 1-year SUMO commutes (analysis/personal_year.py, app mirror, 3 datasets):
        // p < 1e-4 alarms ~2.5x per driver-year on usual commutes and catches ~60% of
        // tired/rushed days; p < 1e-3 ~6.9x per year, ~70%.
        const val P_WATCH = 1e-3
        const val P_ALERT = 1e-4
        // Drift (same evaluation, H checked on all five datasets pooled): ~0.3-0.7 notifications per
        // driver-year when nothing changed; fast, slow (2-3 months) and transient changes caught.
        const val REF_DAYS = 20
        const val LAG_GAP = 20
        const val LAG_LEN = 60
        const val DAY_Z_CAP = 3.0
        const val CUSUM_K = 0.5
        const val CUSUM_H = 10.0
        const val ROAD_SKIP_RATIO = 0.7  // snow days: 0.4-0.7 of the usual expectation, clear/rain >= 0.8
        const val RECENT_DAYS = 10     // a change is reported while it alarmed within this many commute days
        const val Z_CAP = 4.0          // one heavy-tailed feature can't decide alone
        const val MAX_DECEL_CAP = 10.0 // m/s², physical limit (sensor spikes)

        // median within-driver variances of the features on ordinary commutes (1-year SUMO,
        // route x window groups); used while a driver has few past commutes
        const val POP_VAR_MODERATE = 0.10
        const val POP_VAR_ACCEL_RMS = 0.014
        const val POP_VAR_MAX_DECEL = 0.36

        private fun gentleRoad(t: TripSummary): Boolean {
            val usual = t.eUsual ?: return false
            val today = t.eToday ?: return false
            return usual > 0 && today / usual < ROAD_SKIP_RATIO
        }

        fun windowOf(hour: Int) = when (hour) {
            in 5..11 -> "am"
            in 15..21 -> "pm"
            else -> "other"
        }

        /**
         * P(this many hard brakes or more | expected mu), with the driver's own day-to-day spread:
         * variance/mean of past counts, shrunk toward 1 (Poisson) by [N0_DISP] trips. Drivers who
         * brake hard often do it very regularly (ratio ~0.3 in SUMO), so a rise Poisson calls
         * ordinary can be unusual for them. Normal approximation when mu >= 1, else Poisson.
         * Mirrors analysis/personal.py _dispersed_p.
         */
        fun hardBrakeUpper(k: Int, mu: Double, past: List<Int>): Double {
            if (mu < 1.0) return poissonUpper(k, mu)
            val n = past.size
            val m = past.average()
            val v = if (n > 1) past.sumOf { (it - m) * (it - m) } / (n - 1) else 0.0
            val phi = ((n * v / maxOf(m, 0.2) + N0_DISP) / (n + N0_DISP)).coerceIn(0.25, 3.0)
            return normalUpper((k - 0.5 - mu) / sqrt(phi * mu + 0.05)).coerceIn(1e-300, 1.0)
        }

        /** P(X >= k), X ~ Poisson(mu). */
        fun poissonUpper(k: Int, mu: Double): Double {
            if (k <= 0) return 1.0
            var term = exp(-mu)
            var cdf = term
            for (i in 1 until k) {
                term *= mu / i
                cdf += term
            }
            return (1.0 - cdf).coerceIn(1e-300, 1.0)
        }

        private fun shrunkVar(xs: List<Double>, popVar: Double): Double {
            val n = xs.size
            val m = xs.average()
            val v = if (n > 1) xs.sumOf { (it - m) * (it - m) } / n else popVar
            return (n * v + N0_FEATURE * popVar) / (n + N0_FEATURE)
        }

        fun z(x: Double, history: List<Double>, popVar: Double): Double =
            (x - history.average()) / sqrt(shrunkVar(history, popVar).coerceAtLeast(1e-9))

        /**
         * One-sided upper p-value of x vs own history: predictive Student t (few past trips ->
         * mean and variance are uncertain), z capped at [Z_CAP].
         */
        fun zUpper(x: Double, history: List<Double>, popVar: Double): Double {
            val n = history.size
            val v = shrunkVar(history, popVar).coerceAtLeast(1e-9) * (1 + 1.0 / n.coerceAtLeast(1))
            val zt = ((x - history.average()) / sqrt(v)).coerceAtMost(Z_CAP)
            return studentUpper(zt, n + N0_FEATURE)
        }

        fun normalUpper(z: Double): Double = 0.5 * erfc(z / sqrt(2.0))

        /** P(T > t), T ~ Student t with [df] degrees of freedom. */
        fun studentUpper(t: Double, df: Double): Double {
            val tail = 0.5 * incompleteBeta(df / (df + t * t), df / 2, 0.5)
            return if (t >= 0) tail else 1.0 - tail
        }

        /** Regularized incomplete beta I_x(a, b) (Numerical Recipes betai/betacf). */
        private fun incompleteBeta(x: Double, a: Double, b: Double): Double {
            if (x <= 0.0) return 0.0
            if (x >= 1.0) return 1.0
            val bt = exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * ln(x) + b * ln(1 - x))
            return if (x < (a + 1) / (a + b + 2)) bt * betaCf(x, a, b) / a
            else 1.0 - bt * betaCf(1 - x, b, a) / b
        }

        private fun betaCf(x: Double, a: Double, b: Double): Double {
            val tiny = 1e-30
            var c = 1.0
            var d = 1.0 - (a + b) * x / (a + 1)
            if (kotlin.math.abs(d) < tiny) d = tiny
            d = 1 / d
            var h = d
            for (m in 1..200) {
                val m2 = 2 * m
                var aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
                d = 1 + aa * d; if (kotlin.math.abs(d) < tiny) d = tiny
                c = 1 + aa / c; if (kotlin.math.abs(c) < tiny) c = tiny
                d = 1 / d; h *= d * c
                aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
                d = 1 + aa * d; if (kotlin.math.abs(d) < tiny) d = tiny
                c = 1 + aa / c; if (kotlin.math.abs(c) < tiny) c = tiny
                d = 1 / d
                val del = d * c
                h *= del
                if (kotlin.math.abs(del - 1) < 1e-12) break
            }
            return h
        }

        /** ln Γ(x), Lanczos approximation. */
        private fun lnGamma(x: Double): Double {
            val cof = doubleArrayOf(76.18009172947146, -86.50532032941677, 24.01409824083091,
                -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)
            var y = x
            val tmp = x + 5.5 - (x + 0.5) * ln(x + 5.5)
            var ser = 1.000000000190015
            for (c in cof) { y += 1; ser += c / y }
            return -tmp + ln(2.5066282746310005 * ser / x)
        }

        /** Fisher's method; chi-square survival with 2k degrees of freedom (closed form). */
        fun fisher(ps: List<Double>): Double {
            val x = -2 * ps.sumOf { ln(it.coerceIn(1e-300, 1.0)) }
            var term = 1.0
            var sum = 1.0
            for (i in 1 until ps.size) {
                term *= (x / 2) / i
                sum += term
            }
            return (exp(-x / 2) * sum).coerceIn(0.0, 1.0)
        }

        /** Complementary error function (Numerical Recipes erfcc, |error| < 1.2e-7). */
        private fun erfc(x: Double): Double {
            val z = kotlin.math.abs(x)
            val t = 1.0 / (1.0 + 0.5 * z)
            val r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
                t * (-0.82215223 + t * 0.17087277)))))))))
            return if (x >= 0) r else 2.0 - r
        }
    }
}
