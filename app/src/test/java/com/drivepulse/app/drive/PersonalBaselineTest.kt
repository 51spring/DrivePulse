package com.drivepulse.app.drive

import com.drivepulse.app.drive.PersonalBaseline.Level
import com.drivepulse.app.drive.PersonalBaseline.TripSummary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files
import java.time.LocalDate

class PersonalBaselineTest {

    private fun trip(day: Int, hard: Int, moderate: Int = 3, accel: Double = 0.8, maxDecel: Double = 2.8,
                     eToday: Double = 0.4, window: String = "am", route: String = "A>B") =
        TripSummary(LocalDate.of(2026, 1, 1).plusDays(day.toLong()).toString(), window, route, 3.0, hard, moderate,
            accel, maxDecel, 0.4, eToday)

    /** Usual commute with small day-to-day variation. */
    private fun usual(day: Int, route: String = "A>B") =
        trip(day, if (day % 4 == 0) 1 else 0, moderate = 3 + day % 2, accel = 0.8 + (day % 3) * 0.02,
            maxDecel = 2.8 + (day % 5) * 0.1, route = route)

    private fun store(): PersonalBaseline =
        PersonalBaseline(File(Files.createTempDirectory("pb").toFile(), "history.csv"))

    @Test
    fun statisticsMatchReferenceValues() {
        assertEquals(0.2642411, PersonalBaseline.poissonUpper(2, 1.0), 1e-6)   // 1 - e^-1 (1 + 1)
        assertEquals(0.0227501, PersonalBaseline.normalUpper(2.0), 1e-6)
        assertEquals(0.05, PersonalBaseline.fisher(listOf(0.05)), 1e-6)           // single p unchanged
        assertEquals(0.0366941, PersonalBaseline.studentUpper(2.0, 10.0), 1e-6)  // scipy t.sf(2, 10)
        assertEquals(0.5, PersonalBaseline.studentUpper(0.0, 6.0), 1e-9)
        assertEquals(0.9633059, PersonalBaseline.studentUpper(-2.0, 10.0), 1e-6)
        // a regular driver (4 +-0.6 hard brakes a day): 7 is unusual for them, not under Poisson (0.11)
        val regular = listOf(4, 4, 3, 5, 4, 4, 4, 5, 3, 4)
        assertEquals(0.0486086, PersonalBaseline.hardBrakeUpper(7, 4.0, regular), 1e-5)   // analysis/personal.py
        assertEquals(PersonalBaseline.poissonUpper(2, 0.5), PersonalBaseline.hardBrakeUpper(2, 0.5, regular), 1e-12)
    }

    @Test
    fun learnsBeforeJudging() {
        val r = store().evaluate(trip(5, 0), listOf(trip(1, 0), trip(2, 1)))
        assertEquals(Level.LEARNING, r.level)
    }

    @Test
    fun tiredAndRushedDaysGetTheirOwnWords() {
        val history = (1..20).map { usual(it) }
        val tired = store().evaluate(trip(21, 4, moderate = 12, accel = 0.86, maxDecel = 5.0), history)
        assertEquals(PersonalBaseline.Cause.TIRED, tired.cause)
        assertTrue(tired.day.contains("피곤하면"))
        val rushed = store().evaluate(trip(21, 3, moderate = 5, accel = 1.15, maxDecel = 5.5), history)
        assertTrue(rushed.level == Level.ALERT || rushed.level == Level.WATCH)
        assertEquals(PersonalBaseline.Cause.RUSHED, rushed.cause)
        assertNull(store().evaluate(usual(21), history).cause)
    }

    @Test
    fun ordinaryDayIsNormalAndTiredDayIsFlagged() {
        val history = (1..10).map { trip(it, if (it % 3 == 0) 1 else 0) }
        val b = store()
        assertEquals(Level.NORMAL, b.evaluate(trip(11, 0), history).level)
        val tired = b.evaluate(trip(11, 4, moderate = 9, accel = 1.05, maxDecel = 5.5), history)
        assertEquals(Level.ALERT, tired.level)
        assertTrue(tired.day.contains("평소와 많이 달라요"))
    }

    @Test
    fun badRoadDayIsNotBlamedOnTheDriver() {
        val history = (1..10).map { trip(it, if (it % 3 == 0) 1 else 0) }
        // 3 hard brakes, but other cars on the same road also braked ~8x more today
        val r = store().evaluate(trip(11, 3, eToday = 3.2), history)
        assertTrue(r.level == Level.NORMAL || r.level == Level.WATCH)
    }

    @Test
    fun gradualChangeRaisesDriftMessage() {
        val steady = (1..40).map { usual(it) }
        assertNull(store().evaluate(usual(41), steady).drift)
        val drifting = (1..25).map { usual(it) } +
            (26..40).map { trip(it, 3, moderate = 8, accel = 1.0, maxDecel = 4.5) }
        val r = store().evaluate(trip(41, 3, moderate = 8, accel = 1.0, maxDecel = 4.5), drifting)
        assertNotNull(r.drift)
        assertTrue(r.drift!!.contains("2026-01-2"))   // dated near the start of the change (day 26)
    }

    @Test
    fun okAnswerStartsTheLongTermCheckOver() {
        val b = store()
        val history = (1..25).map { usual(it) } +
            (26..40).map { trip(it, 3, moderate = 8, accel = 1.0, maxDecel = 4.5) }
        val today = trip(41, 3, moderate = 8, accel = 1.0, maxDecel = 4.5)
        val r = b.evaluate(today, history)
        assertNotNull(r.drift)
        b.acknowledge(r.driftAck!!)
        assertEquals(today.date, b.acknowledged()["am" to "A>B"])
        // the new way of driving is the new normal: no message, and the check restarts after today
        val later = history + today + (42..50).map { trip(it, 3, moderate = 8, accel = 1.0, maxDecel = 4.5) }
        val next = b.evaluate(trip(51, 3, moderate = 8, accel = 1.0, maxDecel = 4.5), later)
        assertNull(next.drift)
        assertNull(next.driftAck)
    }

    @Test
    fun oneBadDayIsNotADrift() {
        val history = (1..40).map { if (it == 35) trip(it, 6, moderate = 12, accel = 1.2, maxDecel = 7.0) else usual(it) }
        assertNull(store().evaluate(usual(41), history).drift)
    }

    @Test
    fun driftMessageClearsAfterAChangeEnds() {
        val history = (1..25).map { usual(it) } +
            (26..40).map { trip(it, 3, moderate = 8, accel = 1.0, maxDecel = 4.5) } +
            (41..80).map { usual(it) }
        assertNull(store().evaluate(usual(81), history).drift)
    }

    @Test
    fun snowyReferenceDaysAreLeftOut() {
        // a snowy month at the start: everyone drives slowly (road expectation half of usual,
        // fewer brakes); afterwards ordinary days must not look like a change
        val snow = { d: Int -> trip(d, 0, moderate = 1, accel = 0.6, maxDecel = 2.0, eToday = 0.2) }
        val history = (1..30).map { if (it % 3 != 0) snow(it) else usual(it) } + (31..55).map { usual(it) }
        assertNull(store().evaluate(usual(56), history).drift)
        val withoutSkip = history.map { it.copy(eUsual = null) }   // no server data: nothing left out
        assertNotNull(store().evaluate(usual(56), withoutSkip).drift)
    }

    @Test
    fun newRouteStartsItsOwnHistory() {
        val history = (1..30).map { usual(it) }
        val r = store().evaluate(usual(31, route = "C>D"), history + listOf(usual(30, route = "C>D")))
        assertEquals(Level.LEARNING, r.level)
        assertTrue(r.day.contains("새 출근길 경로"))
    }

    @Test
    fun demoHistoriesGiveTheirStories() {
        val load = { name: String ->
            File("src/main/assets/$name").readLines().drop(1).mapNotNull { TripSummary.fromCsv(it) }
        }
        val usual = load("demo_history_gangnam.csv")
        val drifting = load("demo_history_drift_gangnam.csv")
        // the tired demo commute (assets/demo_commute_gangnam.csv as the app summarises it)
        val tired = { date: String -> usual.last().copy(date = date, hardBrakes = 5, moderateBrakes = 13,
            accelRms = 0.68, maxDecel = 4.4) }
        val oneDay = store().evaluate(tired("2026-12-01"), usual)
        assertEquals(Level.ALERT, oneDay.level)
        assertNull(oneDay.drift)
        val weeks = store().evaluate(tired("2026-12-01"), drifting)
        assertNotNull(weeks.drift)
    }

    @Test
    fun csvRoundTrip() {
        val b = store()
        b.append(trip(1, 2))
        b.append(trip(2, 0, window = "pm"))
        assertEquals(listOf(trip(1, 2), trip(2, 0, window = "pm")), b.history())
    }
}
