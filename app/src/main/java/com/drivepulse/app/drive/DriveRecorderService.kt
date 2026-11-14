package com.drivepulse.app.drive

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import com.drivepulse.app.MainActivity
import com.drivepulse.app.R

/**
 * Foreground service so recording keeps going while driving, even when the
 * car's UX restrictions hide or stop the activity.
 */
class DriveRecorderService : Service() {

    private var recorder: DriveRecorder? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        startForeground(NOTIFICATION_ID, buildNotification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
        if (recorder == null) {
            val demo = intent?.getStringExtra(EXTRA_DEMO)?.let { runCatching { DemoTrip.Kind.valueOf(it) }.getOrNull() }
            recorder = DriveRecorder(applicationContext).also { it.start(demo) }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        recorder?.shutdown()
        recorder = null
        super.onDestroy()
    }

    private fun buildNotification(): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "주행 기록", NotificationManager.IMPORTANCE_LOW)
        )
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("주행 데이터 기록 중")
            .setContentText("차속·브레이크·위치를 기록하고 급정거를 감지합니다")
            .setContentIntent(open)
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val CHANNEL_ID = "drive_recording"
        private const val NOTIFICATION_ID = 1
        private const val ACTION_STOP = "com.drivepulse.app.STOP_RECORDING"
        private const val EXTRA_DEMO = "demo"

        /** @param demo demo trip to replay (emulator), or null to record the real drive. */
        fun start(context: Context, demo: DemoTrip.Kind? = null) {
            context.startForegroundService(
                Intent(context, DriveRecorderService::class.java).putExtra(EXTRA_DEMO, demo?.name)
            )
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, DriveRecorderService::class.java).setAction(ACTION_STOP)
            )
        }
    }
}
