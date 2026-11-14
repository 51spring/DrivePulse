package com.drivepulse.app.drive

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Talks to the DrivePulse server (server/app.py). Blocking; call from a worker thread.
 * On the emulator the host PC is reachable at 10.0.2.2.
 */
class ServerClient(private val baseUrl: String) {

    data class Alert(
        val segmentId: String,
        val name: String,
        val windowStart: String,
        val observed: Int,
        val expected: Double,
        val ratio: Double,
    ) {
        /** "평소의 3.2배", or counts when the usual level is near zero and the ratio explodes. */
        val spikeText: String get() = when {
            ratio < 10 -> "평소의 %.1f배".format(ratio)
            expected < 0.5 -> "급정거 ${observed}회, 평소엔 거의 없음"
            else -> "급정거 ${observed}회, 평소 %.0f회".format(expected)
        }
    }

    /** Uploads traversals; returns how many the server stored. */
    fun upload(traversals: List<TraversalTracker.Traversal>): Int {
        val arr = JSONArray()
        traversals.forEach {
            arr.put(JSONObject().apply {
                put("segment_id", it.segmentId)
                put("date", it.date)
                put("hour", it.hour)
                put("minute", it.minute)
                put("hard_brakes", it.hardBrakes)
            })
        }
        // Anonymous: no driver or trip ID, so the server can't link passes into a car's route.
        val body = JSONObject().put("traversals", arr)
        val res = request("POST", "/api/traversals", body.toString())
        return JSONObject(res).optInt("stored", 0)
    }

    /**
     * Expected hard brakes on the given segment passes: usually, and given other cars today.
     * Only segment IDs and times are sent; the driver's own history stays in the car.
     */
    fun tripContext(traversals: List<TraversalTracker.Traversal>): Pair<Double?, Double?> {
        val arr = JSONArray()
        traversals.forEach {
            arr.put(JSONObject().put("segment_id", it.segmentId).put("date", it.date)
                .put("hour", it.hour).put("minute", it.minute))
        }
        val res = JSONObject(request("POST", "/api/trip_context", JSONObject().put("traversals", arr).toString()))
        val usual = if (res.isNull("e_usual")) null else res.getDouble("e_usual")
        val today = if (res.isNull("e_today")) null else res.getDouble("e_today")
        return Pair(usual, today)
    }

    fun alerts(date: String, hour: Int, minute: Int): List<Alert> {
        val res = JSONObject(request("GET", "/api/alerts?date=$date&hour=$hour&minute=$minute", null))
        val arr = res.optJSONArray("alerts") ?: return emptyList()
        return (0 until arr.length()).map { i ->
            val a = arr.getJSONObject(i)
            Alert(
                segmentId = a.getString("segment_id"),
                name = a.optString("name"),
                windowStart = a.optString("window_start"),
                observed = a.optInt("observed"),
                expected = a.optDouble("expected"),
                ratio = a.optDouble("ratio"),
            )
        }
    }

    private fun request(method: String, path: String, body: String?): String {
        val conn = URL(baseUrl.trimEnd('/') + path).openConnection() as HttpURLConnection
        try {
            conn.requestMethod = method
            conn.connectTimeout = 3000
            conn.readTimeout = 5000
            if (body != null) {
                conn.doOutput = true
                conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                conn.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
            }
            val code = conn.responseCode
            val text = (if (code in 200..299) conn.inputStream else conn.errorStream)
                ?.bufferedReader(Charsets.UTF_8)?.use { it.readText() } ?: ""
            if (code !in 200..299) throw java.io.IOException("HTTP $code: $text")
            return text
        } finally {
            conn.disconnect()
        }
    }
}
