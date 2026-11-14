package com.drivepulse.app.drive

import android.annotation.SuppressLint
import android.car.Car
import android.car.VehiclePropertyIds
import android.car.hardware.CarPropertyValue
import android.car.hardware.property.CarPropertyManager
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import ai.pleos.playground.vehicle.Vehicle
import ai.pleos.playground.vehicle.constant.control.VehicleGear
import ai.pleos.playground.vehicle.listener.BrakePedalStatusListener
import ai.pleos.playground.vehicle.listener.GearSelectionListener
import ai.pleos.playground.vehicle.listener.SteeringWheelAngleListener
import com.drivepulse.app.BuildConfig
import java.io.BufferedWriter
import java.io.File
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale
import java.util.UUID

/**
 * Collects driving signals, detects hard brakes, matches the car to road segments and
 * exchanges data with the DrivePulse server.
 *
 * Signals
 *  - speed: AAOS PERF_VEHICLE_SPEED via CarPropertyManager (timestamped, fixed rate)
 *  - brake pedal / gear / steering angle: Pleos Vehicle SDK (change events)
 *  - location: Android LocationManager -> SegmentMap (same segment IDs as SUMO/server)
 *
 * Output
 *  - filesDir/drive_logs/<session>_samples.csv : every speed sample with the other signals
 *  - filesDir/drive_logs/hard_brakes.csv       : one row per hard brake (all sessions)
 *  - server: traversals (segment passes with hard-brake counts) every few seconds
 *  - alerts: segments ahead whose hard braking is spiking right now (from the server)
 *
 * Demo mode replays a SUMO drive through Gangnam (see [DemoTrip.Kind]) with a simulated
 * clock starting at 07:10 today, because the Pleos emulator can't move.
 *
 * All state is touched only on the recorder thread; network calls run on their own thread.
 */
class DriveRecorder(private val context: Context) {

    private val thread = HandlerThread("drive-recorder").apply { start() }
    private val handler = Handler(thread.looper)
    private val netThread = HandlerThread("drive-net").apply { start() }
    private val net = Handler(netThread.looper)
    private val zone = ZoneId.of("Asia/Seoul")
    private val tsFormat = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS").withZone(zone)
    private val sessionFormat = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss").withZone(zone)

    private val detector = HardBrakeDetector()
    private val segmentMap: SegmentMap? by lazy { SegmentMap.load(context) }
    private val server = ServerClient(BuildConfig.SERVER_URL)

    private var car: Car? = null
    private var propertyManager: CarPropertyManager? = null
    private var vehicle: Vehicle? = null
    private var locationManager: LocationManager? = null

    private var sessionId = ""
    private var samplesOut: BufferedWriter? = null
    private var eventsOut: BufferedWriter? = null
    private var linesSinceFlush = 0
    private var tracker: TraversalTracker? = null

    /** Added to wall-clock time; non-zero in simulation (simulated 07:10 start). */
    private var clockOffsetMs = 0L

    // Latest value of each signal.
    private var speedMps: Float? = null
    private var brakePressed: Boolean? = null
    private var lastBrakeOnNanos = Long.MIN_VALUE
    private var gear: String? = null
    private var steeringDeg: Float? = null
    private var location: Location? = null
    private var headingDeg: Double? = null
    private var geohash: String? = null
    private var segment: SegmentMap.Segment? = null

    private var lastSpeedNanos = 0L
    private var rateHz: Float? = null
    private var sampleCount = 0L
    private var distanceM = 0.0
    private var hardBrakeCount = 0
    private var lastEventText: String? = null
    private var lastPublishNanos = 0L
    private val messages = ArrayDeque<String>()
    private var recording = false

    // Trip summary for the personal commute baseline (kept on the device). Demos use their own
    // history file so they never mix with the real commutes.
    private val realHistory by lazy { PersonalBaseline(File(context.filesDir, "personal/commutes.csv")) }
    private var personal: PersonalBaseline? = null
    private val moderateDetector = HardBrakeDetector(startDecelMps2 = 2.0f, endDecelMps2 = 1.0f)
    private var moderateCount = 0
    private var accelSqSum = 0.0
    private var accelN = 0
    private var maxDecel = 0f
    private var tripStartWall: Instant? = null
    private var firstSegmentId: String? = null
    private var lastSegmentId: String? = null
    private var tripFinished = false
    /** False for the segment-warning demo: another car's drive, not this driver's commute. */
    private var comparePersonal = true
    private var tripReport: String? = null
    private var tripLevel: String? = null
    private var driftAck: PersonalBaseline.DriftAck? = null
    private var historyFile: String? = null

    // Server state
    private val pendingUpload = ArrayList<TraversalTracker.Traversal>()
    private var uploadedCount = 0
    private var serverOk: Boolean? = null
    private var alerts: List<ServerClient.Alert> = emptyList()
    private var alertText: String? = null

    /**
     * @param demo replay this SUMO demo trip instead of real speed/location (null = real
     *   drive). Used on the emulator, where speed can't be changed (no Virtual Console card,
     *   no VHAL injection on user builds). Brake/gear/steering stay real.
     */
    fun start(demo: DemoTrip.Kind? = null) = handler.post { startInternal(demo) }

    fun stop() = handler.post { stopInternal() }

    /** Stops and quits the worker threads. */
    fun shutdown() {
        handler.post { stopInternal() }
        // Let the end-of-trip report (one server round trip) finish before the threads stop.
        handler.postDelayed({
            thread.quitSafely()
            net.post { netThread.quitSafely() }
        }, 8_000)
    }

    // ---------------------------------------------------------------- lifecycle

    private fun startInternal(kind: DemoTrip.Kind?) {
        if (recording) return
        recording = true
        resetSessionState()
        val simulate = kind != null
        val demo = kind?.let { DemoTrip.load(context, it) }
        comparePersonal = kind == null || kind.history != null
        personal = if (kind?.history != null)
            PersonalBaseline(File(context.filesDir, "personal/demo_${kind.name.lowercase()}.csv"))
        else realHistory
        clockOffsetMs = if (demo != null) simulatedClockOffset(demo.startMinuteOfDay) else 0L
        // Random suffix: in simulation every run starts at the same simulated 07:10:00.
        sessionId = (if (simulate) "sim_" else "") + sessionFormat.format(now()) + "_" +
            UUID.randomUUID().toString().take(4)
        tracker = TraversalTracker(sessionId, zone)
        openFiles()
        when {
            demo != null -> {
                seedDemoHistory(demo.kind)
                startDemoReplay(demo)
            }
            simulate -> startScriptedSimulation()
            else -> { startSpeed(); startLocation() }
        }
        startVehicleSdk()
        note(if (segmentMap != null) "구간 지도: ${segmentMap!!.area} (${segmentMap!!.segments.size}개 구간)"
             else "구간 지도 없음: geohash로 대신 기록")
        note("기록 시작: $sessionId  서버 ${BuildConfig.SERVER_URL}")
        handler.postDelayed(serverTick, SERVER_TICK_MS)
        publish(force = true)
    }

    private fun stopInternal() {
        if (!recording) return
        recording = false
        handler.removeCallbacks(serverTick)
        stopSimulation()
        propertyManager?.unregisterCallback(speedCallback)
        propertyManager = null
        car?.disconnect()
        car = null
        vehicle?.let { v ->
            runCatching {
                v.getBrake().unregisterBrakePedalStatus(brakeListener)
                v.getGear().unregisterGearSelection(gearListener)
                v.getSteeringWheel().unregisterSteeringWheelAngle(steeringListener)
            }
            v.release()
        }
        vehicle = null
        locationManager?.removeUpdates(locationListener)
        locationManager = null
        finishTrip()
        tracker?.close()?.let { pendingUpload += it }
        flushUploads()
        samplesOut?.close(); samplesOut = null
        eventsOut?.close(); eventsOut = null
        note("기록 정지: 샘플 ${sampleCount}개, 급정거 ${hardBrakeCount}회")
        publish(force = true)
    }

    private fun resetSessionState() {
        detector.reset()
        speedMps = null; brakePressed = null; lastBrakeOnNanos = Long.MIN_VALUE
        gear = null; steeringDeg = null; location = null; headingDeg = null; geohash = null; segment = null
        lastSpeedNanos = 0L; rateHz = null; sampleCount = 0L; distanceM = 0.0
        hardBrakeCount = 0; lastEventText = null; messages.clear()
        pendingUpload.clear(); uploadedCount = 0; serverOk = null; alerts = emptyList(); alertText = null
        moderateDetector.reset(); moderateCount = 0; accelSqSum = 0.0; accelN = 0; maxDecel = 0f
        tripStartWall = null; firstSegmentId = null; lastSegmentId = null
        tripFinished = false; tripReport = null; tripLevel = null; driftAck = null; historyFile = null
    }

    // Internal storage: AAOS runs apps as a secondary user (e.g. user 10) whose external
    // app dir is not readable from adb on user builds. Pull with:
    //   adb exec-out run-as com.drivepulse.app --user 10 cat files/drive_logs/hard_brakes.csv
    private fun logDir(): File =
        File(context.filesDir, "drive_logs").apply { mkdirs() }

    private fun openFiles() {
        val dir = logDir()
        samplesOut = File(dir, "${sessionId}_samples.csv").bufferedWriter().apply {
            appendLine("elapsed_ms,wall_time,speed_kmh,accel_mps2,brake,gear,steering_deg,lat,lon,geohash,segment_id")
        }
        val eventsFile = File(dir, "hard_brakes.csv")
        val isNew = !eventsFile.exists()
        eventsOut = java.io.FileWriter(eventsFile, true).buffered().apply {
            if (isNew) {
                appendLine(
                    "session_id,start_time,end_time,date,hour,minute,segment_id,segment_name,geohash,lat,lon," +
                        "start_kmh,end_kmh,peak_decel_mps2,mean_decel_mps2,duration_s,brake_pressed,gear"
                )
                flush()
            }
        }
    }

    // ---------------------------------------------------------------- simulation

    private var simulationToken: Any? = null

    /** Offset that makes "now" read as today at [minuteOfDay] (KST). */
    private fun simulatedClockOffset(minuteOfDay: Int): Long {
        val target = LocalDate.now(zone).atStartOfDay(zone).plusMinutes(minuteOfDay.toLong()).toInstant()
        return target.toEpochMilli() - System.currentTimeMillis()
    }

    /** Replays the SUMO trip: speed into the detector, positions into map matching. */
    private fun startDemoReplay(demo: DemoTrip) {
        val token = Any()
        simulationToken = token
        val t0Nanos = SystemClock.elapsedRealtimeNanos()
        val t0Up = SystemClock.uptimeMillis()
        demo.samples.forEachIndexed { i, s ->
            handler.postAtTime({
                if (simulationToken !== token) return@postAtTime
                val tNanos = t0Nanos + (s.tSec * 1e9).toLong()
                onLocation(Location("demo").apply {
                    latitude = s.lat; longitude = s.lon; speed = s.speedMps; bearing = s.headingDeg
                    elapsedRealtimeNanos = tNanos
                })
                onSpeed(tNanos, s.speedMps)
                if (i == demo.samples.lastIndex) {
                    note("데모 주행 종료 (${demo.kind.label}, ${s.tSec.toInt()}초)")
                    finishTrip()
                }
            }, token, t0Up + (s.tSec * 1000).toLong())
        }
        note("데모 주행: ${demo.kind.label}, ${demo.area} SUMO 경로 ${demo.samples.last().tSec.toInt()}초, 시뮬레이션 시각 07:10 시작")
    }

    /**
     * Fallback when no demo asset exists. Scripted drive at 10 Hz: two hard brakes
     * (60→20 at -6 m/s², 50→0 at -7 m/s²) and one gentle stop (-1.5 m/s²) that must NOT count.
     */
    private fun startScriptedSimulation() {
        val kmh = { v: Float -> v / 3.6f }
        val profile = mutableListOf<Float>()
        fun hold(v: Float, sec: Float) = repeat((sec * SIM_HZ).toInt()) { profile += v }
        fun ramp(to: Float, accel: Float) {
            var v = profile.lastOrNull() ?: 0f
            val dv = accel / SIM_HZ
            while (if (dv < 0) v + dv > to else v + dv < to) { v += dv; profile += v }
            profile += to
        }
        hold(0f, 2f)
        ramp(kmh(60f), 2.5f); hold(kmh(60f), 4f)
        ramp(kmh(20f), -6f); hold(kmh(20f), 3f)          // hard brake #1
        ramp(kmh(50f), 2f); hold(kmh(50f), 3f)
        ramp(0f, -1.5f); hold(0f, 2f)                     // gentle stop, not an event
        ramp(kmh(50f), 2.5f); hold(kmh(50f), 3f)
        ramp(0f, -7f); hold(0f, 3f)                       // hard brake #2

        val token = Any()
        simulationToken = token
        val stepNanos = 1_000_000_000L / SIM_HZ
        val t0 = SystemClock.elapsedRealtimeNanos()
        profile.forEachIndexed { i, v ->
            handler.postAtTime({
                if (simulationToken === token) onSpeed(t0 + i * stepNanos, v)
                if (i == profile.lastIndex && simulationToken === token) note("시뮬레이션 종료 (급정거 2회 기대)")
            }, token, SystemClock.uptimeMillis() + i * 1000L / SIM_HZ)
        }
        note("시뮬레이션 주행 시작: ${profile.size / SIM_HZ}초 프로파일")
        startLocation()
    }

    /**
     * Demo only: (re)load the simulated driver's past commutes (assets/<history>_<area>.csv,
     * exported from SUMO) into the demo's own history file, so every run starts the same.
     * The commutes keep their order but are moved to the weekdays ending yesterday, so the
     * demo works on any date.
     */
    private fun seedDemoHistory(kind: DemoTrip.Kind) {
        val store = personal ?: return
        val name = "${kind.history ?: return}_gangnam.csv"
        if (context.assets.list("")?.contains(name) != true) return
        val trips = context.assets.open(name).bufferedReader().useLines { lines ->
            lines.drop(1).mapNotNull { PersonalBaseline.TripSummary.fromCsv(it) }.toList()
        }.sortedBy { it.date }
        val dates = ArrayList<String>()
        var d = LocalDate.now(zone).minusDays(1)
        while (dates.size < trips.size) {
            if (d.dayOfWeek.value <= 5) dates += d.toString()
            d = d.minusDays(1)
        }
        dates.reverse()
        store.clear()
        trips.zip(dates).forEach { (t, date) -> store.append(t.copy(date = date)) }
        note("데모: 시뮬레이션 운전자의 과거 출퇴근 ${trips.size}회 기록을 불러왔어요")
    }

    /** End of a trip: summarise it and compare with the driver's own commute history. */
    private fun finishTrip() {
        if (tripFinished) return
        tripFinished = true
        if (!comparePersonal) {
            tripReport = "전방 구간 경고 데모 주행이라 출퇴근 기록과 비교하지 않았어요."
            publish(force = true)
            return
        }
        val km = distanceM / 1000.0
        if (km < 0.3 || tripStartWall == null) {
            tripReport = "운행 거리가 짧아 평소와 비교하지 않았어요."
            publish(force = true)
            return
        }
        val start = tripStartWall!!.atZone(zone)
        val traversals = tracker?.allTraversals().orEmpty()
        val hard = hardBrakeCount
        val moderate = moderateCount
        val rms = if (accelN > 0) kotlin.math.sqrt(accelSqSum / accelN) else 0.0
        val maxDec = maxDecel.toDouble()
        val route = "${firstSegmentId ?: "?"}>${lastSegmentId ?: "?"}"
        net.post {
            val ctx = runCatching { server.tripContext(traversals) }.getOrNull()
            handler.post {
                val summary = PersonalBaseline.TripSummary(
                    date = start.toLocalDate().toString(), window = PersonalBaseline.windowOf(start.hour),
                    routeKey = route, km = km, hardBrakes = hard, moderateBrakes = moderate, accelRms = rms,
                    maxDecel = maxDec, eUsual = ctx?.first, eToday = ctx?.second,
                )
                val store = personal ?: realHistory
                val report = store.evaluate(summary)
                store.append(summary)
                tripLevel = report.level.name
                driftAck = report.driftAck
                historyFile = store.file.absolutePath
                tripReport = report.day + (report.drift?.let { "\n$it" } ?: "") +
                    (if (ctx == null) "\n(서버 연결이 없어 도로 사정 보정 없이 비교했어요)" else "")
                Log.i(TAG, "TRIP_REPORT ${report.level} ${report.day}")
                publish(force = true)
            }
        }
    }

    private fun stopSimulation() {
        simulationToken?.let { handler.removeCallbacksAndMessages(it) }
        simulationToken = null
    }

    // ---------------------------------------------------------------- sources

    private fun startSpeed() {
        try {
            val c = Car.createCar(context)
            car = c
            val pm = c.getCarManager(Car.PROPERTY_SERVICE) as CarPropertyManager
            propertyManager = pm
            val config = pm.getCarPropertyConfig(VehiclePropertyIds.PERF_VEHICLE_SPEED)
            if (config == null) {
                note("PERF_VEHICLE_SPEED 미지원 또는 권한 없음")
                return
            }
            val rate = config.maxSampleRate.takeIf { it > 0f }?.coerceAtMost(TARGET_SPEED_HZ)
                ?: CarPropertyManager.SENSOR_RATE_FASTEST
            val ok = pm.registerCallback(speedCallback, VehiclePropertyIds.PERF_VEHICLE_SPEED, rate)
            note("차속 구독 ${if (ok) "성공" else "실패"} (요청 ${rate}Hz, 최대 ${config.maxSampleRate}Hz)")
        } catch (e: Exception) {
            note("차속 구독 오류: ${e.message}")
            Log.e(TAG, "speed subscribe failed", e)
        }
    }

    private val speedCallback = object : CarPropertyManager.CarPropertyEventCallback {
        override fun onChangeEvent(value: CarPropertyValue<*>) {
            if (value.status != CarPropertyValue.STATUS_AVAILABLE) return
            val v = (value.value as? Float) ?: return
            val ts = value.timestamp
            handler.post { onSpeed(ts, v) }
        }

        override fun onErrorEvent(propertyId: Int, areaId: Int) {
            handler.post { note("차속 오류 이벤트 prop=$propertyId area=$areaId") }
        }
    }

    private fun startVehicleSdk() {
        try {
            val v = Vehicle(context)
            v.initialize()
            vehicle = v
            v.getBrake().registerBrakePedalStatus(brakeListener)
            v.getGear().registerGearSelection(gearListener)
            v.getSteeringWheel().registerSteeringWheelAngle(steeringListener)
            note("Vehicle SDK 구독: 브레이크, 기어, 조향각")
        } catch (e: Exception) {
            note("Vehicle SDK 오류: ${e.message}")
            Log.e(TAG, "vehicle sdk failed", e)
        }
    }

    private val brakeListener = object : BrakePedalStatusListener {
        override fun onBrakePedalStatusUpdated(status: Boolean?) {
            val now = SystemClock.elapsedRealtimeNanos()
            handler.post {
                brakePressed = status
                if (status == true) lastBrakeOnNanos = now
            }
        }

        override fun onFailed(e: Exception) {
            handler.post { note("브레이크 구독 실패: ${e.message}") }
        }
    }

    private val gearListener = object : GearSelectionListener {
        override fun onSelected(gear: VehicleGear?) {
            handler.post { this@DriveRecorder.gear = gear?.name?.removePrefix("GEAR_") }
        }

        override fun onFailed(e: Exception) {
            handler.post { note("기어 구독 실패: ${e.message}") }
        }
    }

    private val steeringListener = object : SteeringWheelAngleListener {
        override fun onSteeringWheelAngleUpdated(angle: Float?) {
            handler.post { steeringDeg = angle }
        }

        override fun onFailed(e: Exception) {
            handler.post { note("조향각 구독 실패: ${e.message}") }
        }
    }

    @SuppressLint("MissingPermission") // checked by the activity before starting
    private fun startLocation() {
        val lm = context.getSystemService(LocationManager::class.java)
        locationManager = lm
        val provider = listOf(LocationManager.FUSED_PROVIDER, LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
            .firstOrNull { lm.hasProvider(it) && lm.isProviderEnabled(it) }
        if (provider == null) {
            note("사용 가능한 위치 제공자 없음")
            return
        }
        try {
            lm.requestLocationUpdates(provider, 1000L, 0f, locationListener, thread.looper)
            lm.getLastKnownLocation(provider)?.let { onLocation(it) }
            note("위치 구독: $provider")
        } catch (e: SecurityException) {
            note("위치 권한 없음")
        }
    }

    private val locationListener = LocationListener { onLocation(it) }

    private fun onLocation(loc: Location) {
        val prev = location
        location = loc
        geohash = Geohash.encode(loc.latitude, loc.longitude, GEOHASH_PRECISION)
        headingDeg = when {
            loc.hasBearing() && loc.speed > 1f -> loc.bearing.toDouble()
            prev != null && SegmentMap.distanceM(prev.latitude, prev.longitude, loc.latitude, loc.longitude) > 3 ->
                SegmentMap.bearing(prev.latitude, prev.longitude, loc.latitude, loc.longitude)
            else -> headingDeg
        }
        val map = segmentMap
        val matched = map?.match(loc.latitude, loc.longitude, headingDeg, segment?.id)
        if (matched != null) {
            segment = matched.segment
            if (firstSegmentId == null) firstSegmentId = matched.segment.id
            lastSegmentId = matched.segment.id
        }
        // Off-map or unmatched: fall back to the geohash cell so the data is still usable.
        val segmentId = matched?.segment?.id ?: if (map == null) "gh:$geohash" else segment?.id
        val tNanos = if (loc.elapsedRealtimeNanos > 0) loc.elapsedRealtimeNanos else SystemClock.elapsedRealtimeNanos()
        if (segmentId != null && (speedMps ?: loc.speed) > 1f) tracker?.onSegment(segmentId, tNanos, wallTime(tNanos))
    }

    // ---------------------------------------------------------------- processing

    private fun onSpeed(tNanos: Long, speed: Float) {
        if (!recording) return
        if (lastSpeedNanos != 0L && tNanos > lastSpeedNanos) {
            val dt = (tNanos - lastSpeedNanos) / 1e9
            if (dt < 2.0) {
                val prev = speedMps ?: speed
                distanceM += (prev + speed) / 2.0 * dt
                val inst = (1.0 / dt).toFloat()
                rateHz = rateHz?.let { it * 0.9f + inst * 0.1f } ?: inst
                if (prev > 0.5f || speed > 0.5f) {   // standstill in queues would dilute jerkiness
                    val a = (speed - prev) / dt
                    accelSqSum += a * a
                    accelN++
                }
            }
        }
        lastSpeedNanos = tNanos
        speedMps = speed
        sampleCount++
        if (tripStartWall == null && speed > 1f) tripStartWall = wallTime(tNanos)

        if (moderateDetector.onSample(SpeedSample(tNanos, speed)) != null) moderateCount++
        val event = detector.onSample(SpeedSample(tNanos, speed))
        detector.lastAccelMps2?.let { if (-it > maxDecel) maxDecel = -it }
        writeSample(tNanos, speed)
        if (event != null) onHardBrake(event)
        publish()
    }

    private fun writeSample(tNanos: Long, speed: Float) {
        val loc = location
        samplesOut?.appendLine(
            listOf(
                tNanos / 1_000_000,
                tsFormat.format(wallTime(tNanos)),
                fmt(speed * 3.6f),
                detector.lastAccelMps2?.let { fmt(it) } ?: "",
                brakePressed?.let { if (it) 1 else 0 } ?: "",
                gear ?: "",
                steeringDeg?.let { fmt(it) } ?: "",
                loc?.latitude ?: "",
                loc?.longitude ?: "",
                geohash ?: "",
                segment?.id ?: "",
            ).joinToString(",")
        )
        if (++linesSinceFlush >= 50) {
            samplesOut?.flush()
            linesSinceFlush = 0
        }
    }

    private fun onHardBrake(e: HardBrakeEvent) {
        hardBrakeCount++
        val credited = tracker?.onHardBrake(e.startNanos) ?: segment?.id
        val name = credited?.let { segmentMap?.byId?.get(it)?.name }.orEmpty()
        val start = wallTime(e.startNanos).atZone(zone)
        val brake = brakePressed == true || lastBrakeOnNanos >= e.startNanos
        val loc = location
        eventsOut?.apply {
            appendLine(
                listOf(
                    sessionId,
                    tsFormat.format(wallTime(e.startNanos)),
                    tsFormat.format(wallTime(e.endNanos)),
                    start.toLocalDate(), start.hour, start.minute,
                    credited ?: "", "\"$name\"", geohash ?: "", loc?.latitude ?: "", loc?.longitude ?: "",
                    fmt(e.startSpeedMps * 3.6f), fmt(e.endSpeedMps * 3.6f),
                    fmt(e.peakDecelMps2), fmt(e.meanDecelMps2), fmt(e.durationSec),
                    if (brake) 1 else 0, gear ?: "",
                ).joinToString(",")
            )
            flush()
        }
        lastEventText = "%02d:%02d %s · %.0f→%.0f km/h · 최대 %.1f m/s²".format(
            start.hour, start.minute, name.ifEmpty { credited ?: geohash ?: "-" },
            e.startSpeedMps * 3.6f, e.endSpeedMps * 3.6f, e.peakDecelMps2
        )
        Log.i(TAG, "HARD_BRAKE $lastEventText")
        publish(force = true)
    }

    // ---------------------------------------------------------------- server

    /** Every few seconds: upload finished traversals, refresh alerts. */
    private val serverTick: Runnable = object : Runnable {
        override fun run() {
            if (!recording) return
            tracker?.drain()?.let { pendingUpload += it }
            flushUploads()
            refreshAlerts()
            handler.postDelayed(this, SERVER_TICK_MS)
        }
    }

    private fun flushUploads() {
        if (pendingUpload.isEmpty()) return
        val batch = ArrayList(pendingUpload)
        pendingUpload.clear()
        net.post {
            val result = runCatching { server.upload(batch) }
            handler.post {
                result.onSuccess { uploadedCount += it; serverOk = true }
                    .onFailure {
                        serverOk = false
                        if (pendingUpload.size < 5000) pendingUpload.addAll(0, batch)  // retry next tick
                        Log.w(TAG, "upload failed: ${it.message}")
                    }
                publish(force = true)
            }
        }
    }

    private fun refreshAlerts() {
        val t = now().atZone(zone)
        net.post {
            val result = runCatching { server.alerts(t.toLocalDate().toString(), t.hour, t.minute) }
            handler.post {
                result.onSuccess { alerts = it; serverOk = true }.onFailure { serverOk = false }
                updateAlertText()
            }
        }
    }

    /** Shows an alert only for segments the car is on or heading toward. */
    private fun updateAlertText() {
        val loc = location
        val heading = headingDeg
        val map = segmentMap
        val relevant = if (loc != null && heading != null && map != null) {
            val ahead = map.ahead(loc.latitude, loc.longitude, heading)
            alerts.filter { it.segmentId in ahead }
        } else emptyList()
        val a = relevant.maxByOrNull { it.ratio }
        val text = a?.let {
            "⚠ 전방 ${it.name.ifEmpty { "구간" }} 급정거 급증 (${it.spikeText}, ${it.windowStart}~)"
        }
        if (text != alertText) {
            alertText = text
            if (text != null) Log.i(TAG, "ALERT $text")
            publish(force = true)
        }
    }

    // ---------------------------------------------------------------- helpers

    private fun now(): Instant = Instant.ofEpochMilli(System.currentTimeMillis() + clockOffsetMs)

    /** Converts an elapsedRealtimeNanos timestamp to (possibly simulated) wall-clock time. */
    private fun wallTime(tNanos: Long): Instant {
        val ageMs = (SystemClock.elapsedRealtimeNanos() - tNanos) / 1_000_000
        return Instant.ofEpochMilli(System.currentTimeMillis() + clockOffsetMs - ageMs)
    }

    private fun fmt(v: Float) = "%.3f".format(Locale.US, v)

    private fun note(msg: String) {
        Log.i(TAG, msg)
        messages.addLast(msg)
        while (messages.size > 6) messages.removeFirst()
        publish(force = true)
    }

    private fun publish(force: Boolean = false) {
        val nowNanos = SystemClock.elapsedRealtimeNanos()
        if (!force && nowNanos - lastPublishNanos < 200_000_000L) return
        lastPublishNanos = nowNanos
        val loc = location
        DriveStatusBus.publish(
            DriveStatus(
                recording = recording,
                speedKmh = speedMps?.let { it * 3.6f },
                accelMps2 = detector.lastAccelMps2,
                speedRateHz = rateHz,
                brakePressed = brakePressed,
                gear = gear,
                steeringDeg = steeringDeg,
                lat = loc?.latitude,
                lon = loc?.longitude,
                geohash = geohash,
                segmentId = segment?.id,
                segmentName = segment?.name,
                clock = if (recording) DateTimeFormatter.ofPattern("HH:mm:ss").withZone(zone).format(now()) else null,
                sampleCount = sampleCount,
                distanceKm = distanceM / 1000.0,
                hardBrakeCount = hardBrakeCount,
                lastEvent = lastEventText,
                uploadedTraversals = uploadedCount,
                serverOk = serverOk,
                alert = alertText,
                tripReport = tripReport,
                tripLevel = tripLevel,
                driftAck = driftAck,
                historyFile = historyFile,
                logDir = logDir().absolutePath,
                messages = messages.toList(),
            )
        )
    }

    companion object {
        private const val TAG = "DrivePulse"
        private const val TARGET_SPEED_HZ = 10f
        private const val SIM_HZ = 10
        private const val GEOHASH_PRECISION = 7
        private const val SERVER_TICK_MS = 5_000L

        /** Random per-install ID (not linked to a person) used to group a car's uploads. */
        fun driverId(context: Context): String {
            val prefs = context.getSharedPreferences("drivepulse", Context.MODE_PRIVATE)
            return prefs.getString("driver_id", null) ?: UUID.randomUUID().toString().also {
                prefs.edit().putString("driver_id", it).apply()
            }
        }
    }
}
