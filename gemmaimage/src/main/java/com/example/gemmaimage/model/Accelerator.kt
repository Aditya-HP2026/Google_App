package com.example.gemmaimage.model

/**
 * Accelerator backends supported by LiteRT-LM.
 */
enum class Accelerator {
    GPU,
    CPU;

    val displayName: String
        get() = when (this) {
            GPU -> "GPU"
            CPU -> "CPU"
        }
}
