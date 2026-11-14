plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.drivepulse.app"
    useLibrary("android.car")
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.drivepulse.app"
        minSdk = 34
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // DrivePulse server (server/app.py). 10.0.2.2 = host PC as seen from the emulator.
        // Override with -PdrivepulseServer=http://<host>:8000
        val server = (project.findProperty("drivepulseServer") as String?) ?: "http://10.0.2.2:8000"
        buildConfigField("String", "SERVER_URL", "\"$server\"")
    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        release {
            optimization {
                enable = true
                packageScope = setOf("androidx.**", "kotlin.**", "kotlinx.**")
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.pleos.vehicle)
    testImplementation(libs.junit)
    androidTestImplementation(libs.espresso.core)
    androidTestImplementation(libs.ext.junit)
}