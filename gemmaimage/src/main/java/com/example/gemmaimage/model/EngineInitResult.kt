package com.example.gemmaimage.model

/**
 * Result of engine initialization — tells you what actually happened.
 */
data class EngineInitResult(
    /** The backend that was requested. */
    val requestedBackend: Accelerator,
    /** The backend that is actually running. */
    val activeBackend: Accelerator,
    /** Whether a fallback from GPU → CPU occurred. */
    val didFallback: Boolean,
    /** If fallback happened, why. Null if no fallback. */
    val fallbackReason: String? = null,
    /** Error class name if fallback was triggered. */
    val fallbackErrorClass: String? = null,
    /** Time taken to initialize the engine in milliseconds. */
    val initTimeMs: Long
)
