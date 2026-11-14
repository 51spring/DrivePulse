package com.drivepulse.app.drive

/**
 * Standard geohash encoding. Precision 7 ≈ 153 m × 153 m cells, used here as a
 * stand-in for a road segment ID until real map matching is added.
 */
object Geohash {
    private const val BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

    fun encode(lat: Double, lon: Double, precision: Int = 7): String {
        var latLo = -90.0
        var latHi = 90.0
        var lonLo = -180.0
        var lonHi = 180.0
        val out = StringBuilder(precision)
        var bit = 0
        var ch = 0
        var evenBit = true
        while (out.length < precision) {
            if (evenBit) {
                val mid = (lonLo + lonHi) / 2
                if (lon >= mid) { ch = ch or (1 shl (4 - bit)); lonLo = mid } else lonHi = mid
            } else {
                val mid = (latLo + latHi) / 2
                if (lat >= mid) { ch = ch or (1 shl (4 - bit)); latLo = mid } else latHi = mid
            }
            evenBit = !evenBit
            if (++bit == 5) {
                out.append(BASE32[ch])
                bit = 0
                ch = 0
            }
        }
        return out.toString()
    }
}
