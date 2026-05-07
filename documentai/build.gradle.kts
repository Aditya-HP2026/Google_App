plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace  = "com.example.documentai"
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

    // Keep the large vocab / model assets uncompressed so they can be
    // memory-mapped by ONNX Runtime without a full copy.
    packaging {
        jniLibs.keepDebugSymbols += "**/*.so"
        resources.excludes += "**/*.onnx"
        resources.excludes += "**/*.json"
        resources.excludes += "**/*.yml"
    }
    androidResources {
        noCompress += listOf("onnx", "json", "yml")
    }
}

dependencies {
    implementation(libs.onnxruntime.android)
    implementation(libs.androidx.core.ktx)
}
