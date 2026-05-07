plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.example.gemmaimage"
    compileSdk = 36

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("proguard-rules.pro")
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    packaging {
        jniLibs.keepDebugSymbols += "**/*.so"
    }
    androidResources {
        noCompress += listOf("onnx", "onnx_data", "data", "bin", "json", "jinja", "txt", "litertlm")
    }
}

dependencies {
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.11.0-rc1")
    implementation(libs.androidx.core.ktx)
}
