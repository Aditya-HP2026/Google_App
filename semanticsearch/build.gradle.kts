plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.example.semanticsearch"
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

    androidResources {
        noCompress += listOf("onnx", "json")
    }
}

dependencies {
    implementation(libs.onnxruntime.android)
    implementation(libs.androidx.core.ktx)
}
