package com.example.gemmaimage

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.example.gemmaimage.model.Accelerator
import com.example.gemmaimage.model.EngineInitResult
import com.example.gemmaimage.model.GemmaImageResult
import com.example.gemmaimage.model.InferenceMetrics
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * On-device Gemma 4 E2B pipeline using LiteRT-LM with GPU acceleration + CPU fallback.
 *
 * Required model file (2.58 GB, placed once by the user):
 *   Android/data/<package>/files/gemma-4-E2B-it.litertlm
 *
 * Construction is expensive (2–5 s) — create once and reuse.
 * All public methods are blocking; call from a background thread / coroutine.
 */
class GemmaImagePipeline(
    context: Context,
    private val modelFile: File = File(context.getExternalFilesDir(null), MODEL_FILE_NAME),
    private val preferredBackend: Accelerator = Accelerator.GPU
) : AutoCloseable {

    companion object {
        private const val TAG = "GemmaImagePipeline"
        const val MODEL_FILE_NAME = "gemma-4-E2B-it.litertlm"

        val ALL_SUBJECTS = listOf(
            "maths", "physics", "chemistry", "biology", "history",
            "geography", "civics", "science", "social_studies",
            "accounting", "business_studies"
        )

        private fun buildClassifyPrompt(subjects: List<String>): String =
            "You are an academic document classifier. Analyze the content carefully. " +
            "Choose exactly one subject from: ${subjects.joinToString(", ")}. " +
            "Extract 5 to 12 useful keywords. " +
            "Return ONLY valid JSON — no extra text:\n" +
            "{\"subject\":\"chemistry\",\"confidence\":0.95," +
            "\"keywords\":[\"keyword1\",\"keyword2\"],\"evidence\":\"short reason\"}\n" +
            "The confidence must be a float from 0.0 to 1.0."

        private const val SOLVE_PROMPT =
            "You are an expert academic tutor for students in grades 6–12. " +
            "Explain the following question or concept clearly and step-by-step. " +
            "Use simple language. If it involves calculation, show each step."
    }

    private val appContext: Context = context.applicationContext

    /** Returns true if the .litertlm model file exists on disk. */
    val isModelAvailable: Boolean get() = modelFile.exists()

    // Engine state
    private var _engine: Engine? = null
    private var _initResult: EngineInitResult? = null
    private var _runCounter: Int = 0

    /** The result of engine initialization — null until first use. */
    val initResult: EngineInitResult? get() = _initResult

    /** The backend actually in use after initialization. */
    val activeBackend: Accelerator? get() = _initResult?.activeBackend

    // ─── Backend Selection ────────────────────────────────────────────────────

    private fun mapAcceleratorToBackend(accelerator: Accelerator): Backend = when (accelerator) {
        Accelerator.GPU -> Backend.GPU()
        Accelerator.CPU -> Backend.CPU()
    }

    // ─── Engine Initialization with Fallback ──────────────────────────────────

    private fun requireEngine(): Engine {
        _engine?.let { return it }
        check(modelFile.exists()) {
            "Gemma model file not found at ${modelFile.absolutePath}.\n" +
            "Download gemma-4-E2B-it.litertlm from " +
            "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm " +
            "then push it:\n" +
            "  adb push gemma-4-E2B-it.litertlm " +
            "/sdcard/Android/data/com.example.googleapp/files/gemma-4-E2B-it.litertlm"
        }

        Log.i(TAG, "┌─────────────────────────────────────────────────")
        Log.i(TAG, "│ Engine init requested: backend=${preferredBackend.displayName}")
        Log.i(TAG, "│ Model: ${modelFile.name} (${modelFile.length() / 1_048_576} MB)")
        Log.i(TAG, "└─────────────────────────────────────────────────")

        val startTime = System.currentTimeMillis()

        // ── Step 1: Try preferred backend ─────────────────────────────────
        try {
            val engine = createEngine(preferredBackend)
            val elapsed = System.currentTimeMillis() - startTime
            _initResult = EngineInitResult(
                requestedBackend = preferredBackend,
                activeBackend = preferredBackend,
                didFallback = false,
                initTimeMs = elapsed
            )
            logInitSuccess(_initResult!!)
            _engine = engine
            return engine
        } catch (e: Exception) {
            val elapsed = System.currentTimeMillis() - startTime
            Log.w(TAG, "┌─── ${preferredBackend.displayName} INIT FAILED (${elapsed}ms) ───")
            Log.w(TAG, "│ Error: ${classifyError(e)}")
            Log.w(TAG, "│ Class: ${e::class.qualifiedName}")
            Log.w(TAG, "│ Message: ${e.message?.take(200)}")
            Log.w(TAG, "└─── Falling back to CPU... ───")

            // ── Step 2: Fallback to CPU ───────────────────────────────────
            if (preferredBackend != Accelerator.CPU) {
                return tryFallbackCpu(e, startTime)
            } else {
                throw GemmaInitException(
                    "Failed to initialize engine on CPU",
                    preferredBackend, null, e
                )
            }
        }
    }

    private fun tryFallbackCpu(originalError: Exception, overallStartTime: Long): Engine {
        try {
            val engine = createEngine(Accelerator.CPU)
            val totalElapsed = System.currentTimeMillis() - overallStartTime
            _initResult = EngineInitResult(
                requestedBackend = preferredBackend,
                activeBackend = Accelerator.CPU,
                didFallback = true,
                fallbackReason = classifyError(originalError),
                fallbackErrorClass = originalError::class.qualifiedName,
                initTimeMs = totalElapsed
            )
            logInitSuccess(_initResult!!)
            _engine = engine
            return engine
        } catch (cpuError: Exception) {
            val totalElapsed = System.currentTimeMillis() - overallStartTime
            Log.e(TAG, "┌─── CPU FALLBACK ALSO FAILED (${totalElapsed}ms) ───")
            Log.e(TAG, "│ GPU error: ${originalError.message?.take(100)}")
            Log.e(TAG, "│ CPU error: ${cpuError.message?.take(100)}")
            Log.e(TAG, "└─── No backend available ───")
            throw GemmaInitException(
                "All backends failed. GPU: ${classifyError(originalError)}. CPU: ${classifyError(cpuError)}",
                preferredBackend, Accelerator.CPU, cpuError
            )
        }
    }

    private fun createEngine(backend: Accelerator): Engine {
        val liteRtBackend = mapAcceleratorToBackend(backend)
        val config = EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = liteRtBackend,
            visionBackend = if (backend == Accelerator.GPU) Backend.GPU() else Backend.CPU(),
            cacheDir = modelFile.parent
        )
        val engine = Engine(config)
        engine.initialize()
        return engine
    }

    private fun logInitSuccess(result: EngineInitResult) {
        Log.i(TAG, "┌─────────────────────────────────────────────────")
        Log.i(TAG, "│ ✅ Engine initialized successfully")
        Log.i(TAG, "│ Requested:  ${result.requestedBackend.displayName}")
        Log.i(TAG, "│ Active:     ${result.activeBackend.displayName}")
        Log.i(TAG, "│ Fallback:   ${if (result.didFallback) "YES (${result.fallbackReason})" else "NO"}")
        Log.i(TAG, "│ Init time:  ${result.initTimeMs} ms")
        Log.i(TAG, "└─────────────────────────────────────────────────")
    }

    // ─── Error Taxonomy ───────────────────────────────────────────────────────

    private fun classifyError(e: Exception): String {
        val msg = (e.message ?: e.toString()).lowercase()
        return when {
            "unsupported" in msg && "delegate" in msg -> "UNSUPPORTED_DELEGATE"
            "opencl" in msg -> "GPU_NOT_AVAILABLE"
            "gpu" in msg && "not available" in msg -> "GPU_NOT_AVAILABLE"
            "out of memory" in msg || "oom" in msg || "alloc" in msg -> "OUT_OF_MEMORY"
            "incompatible" in msg -> "INCOMPATIBLE_MODEL"
            "model" in msg && "version" in msg -> "INCOMPATIBLE_MODEL"
            "permission" in msg -> "PERMISSION_DENIED"
            "not found" in msg || "no such file" in msg -> "MODEL_NOT_FOUND"
            "timeout" in msg -> "TIMEOUT"
            "unknown model type" in msg -> "UNKNOWN_MODEL_TYPE"
            else -> "UNKNOWN (${e::class.simpleName}: ${msg.take(60)})"
        }
    }

    /** User-friendly message derived from the error taxonomy. */
    fun userFriendlyError(e: Exception): String {
        val taxonomy = classifyError(e)
        return when (taxonomy) {
            "UNSUPPORTED_DELEGATE" -> "This device doesn't support GPU inference. Using CPU instead."
            "GPU_NOT_AVAILABLE" -> "GPU acceleration not available on this device."
            "OUT_OF_MEMORY" -> "Not enough memory to load the model. Close other apps and try again."
            "INCOMPATIBLE_MODEL" -> "Model file is not compatible with this runtime version."
            "PERMISSION_DENIED" -> "Cannot access model file. Check app permissions."
            "MODEL_NOT_FOUND" -> "Model file not found. Push it via ADB first."
            "TIMEOUT" -> "Model loading timed out. Restart the app."
            "UNKNOWN_MODEL_TYPE" -> "Model format is not recognized by LiteRT-LM."
            else -> "Unexpected error: ${e.message?.take(80)}"
        }
    }

    // ─── Inference with Telemetry ─────────────────────────────────────────────

    /**
     * Classifies a [bitmap] image into an academic subject + keywords.
     * Uses vision-enabled session: image + prompt → JSON response.
     */
    fun classify(bitmap: Bitmap, subjects: List<String> = ALL_SUBJECTS): GemmaImageResult {
        val start = System.currentTimeMillis()
        val engine = requireEngine()
        val prompt = buildClassifyPrompt(subjects)
        val jpegBytes = bitmapToJpegBytes(bitmap)

        _runCounter++
        val runNum = _runCounter

        Log.d(TAG, "[Run #$runNum] classify(image) | backend=${activeBackend?.displayName} | prompt=${prompt.length} chars | image=${jpegBytes.size} bytes")

        val config = ConversationConfig(
            samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = 0.1)
        )
        val rawResponse = engine.createConversation(config).use { conv ->
            val response = conv.sendMessage(
                Contents.of(
                    Content.ImageBytes(jpegBytes),
                    Content.Text(prompt)
                )
            )
            response.toString()
        }

        val elapsed = System.currentTimeMillis() - start
        logInferenceMetrics(
            InferenceMetrics(
                backend = activeBackend ?: preferredBackend,
                endToEndMs = elapsed,
                inputSize = prompt.length + jpegBytes.size,
                outputLength = rawResponse.length,
                isVisionCall = true,
                runNumber = runNum
            )
        )

        return GemmaResponseParser.parse(rawResponse, elapsed)
    }

    /**
     * Classifies a text chunk (e.g. from OCR) into an academic subject.
     * Text-only — no image session needed.
     */
    fun classifyText(text: String, subjects: List<String> = ALL_SUBJECTS): GemmaImageResult {
        val start = System.currentTimeMillis()
        val prompt = buildClassifyPrompt(subjects) + "\n\nText to classify:\n" + text.take(2000)

        _runCounter++
        val runNum = _runCounter

        Log.d(TAG, "[Run #$runNum] classifyText | backend=${activeBackend?.displayName} | prompt=${prompt.length} chars")

        val rawResponse = requireEngine().createConversation().use { conv ->
            conv.sendMessage(prompt).toString()
        }

        val elapsed = System.currentTimeMillis() - start
        logInferenceMetrics(
            InferenceMetrics(
                backend = activeBackend ?: preferredBackend,
                endToEndMs = elapsed,
                inputSize = prompt.length,
                outputLength = rawResponse.length,
                isVisionCall = false,
                runNumber = runNum
            )
        )

        return GemmaResponseParser.parse(rawResponse, elapsed)
    }

    /**
     * Explains an academic question using Gemma as a step-by-step tutor.
     */
    fun solveDoubt(question: String): String {
        val prompt = "$SOLVE_PROMPT\n\nQuestion:\n${question.take(2000)}"

        _runCounter++
        Log.d(TAG, "[Run #$_runCounter] solveDoubt | backend=${activeBackend?.displayName}")

        return requireEngine().createConversation().use { conv ->
            conv.sendMessage(prompt).toString()
        }
    }

    /**
     * Parses a raw Gemma JSON string without running inference.
     * Useful for UI testing.
     */
    fun parseGeneratedResponse(rawResponse: String, elapsedMs: Long = 0L): GemmaImageResult =
        GemmaResponseParser.parse(rawResponse, elapsedMs)

    // ─── Telemetry Logging ────────────────────────────────────────────────────

    private fun logInferenceMetrics(metrics: InferenceMetrics) {
        val type = if (metrics.isVisionCall) "VISION" else "TEXT"
        Log.i(TAG, "┌─── Inference #${metrics.runNumber} ($type) ───")
        Log.i(TAG, "│ Backend:     ${metrics.backend.displayName}")
        Log.i(TAG, "│ E2E time:    ${metrics.endToEndMs} ms")
        Log.i(TAG, "│ Input size:  ${metrics.inputSize} bytes")
        Log.i(TAG, "│ Output len:  ${metrics.outputLength} chars")
        Log.i(TAG, "│ Throughput:  ~${if (metrics.endToEndMs > 0) (metrics.outputLength * 1000L / metrics.endToEndMs) else 0} chars/s")
        Log.i(TAG, "└─────────────────────────────────────────────────")
    }

    // ─── Utilities ────────────────────────────────────────────────────────────

    private fun bitmapToJpegBytes(bitmap: Bitmap): ByteArray {
        // Downscale if too large to reduce inference time
        val maxDim = maxOf(bitmap.width, bitmap.height)
        val scaledBitmap = if (maxDim > 768) {
            val scale = 768f / maxDim
            val w = (bitmap.width * scale).toInt().coerceAtLeast(1)
            val h = (bitmap.height * scale).toInt().coerceAtLeast(1)
            Bitmap.createScaledBitmap(bitmap, w, h, true)
        } else {
            bitmap
        }
        val out = ByteArrayOutputStream()
        scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 75, out)
        if (scaledBitmap != bitmap) scaledBitmap.recycle()
        return out.toByteArray()
    }

    override fun close() {
        _engine?.close()
        _engine = null
        _initResult = null
        Log.i(TAG, "Engine closed. Total runs: $_runCounter")
    }
}

/**
 * Thrown when engine initialization fails on all backends.
 */
class GemmaInitException(
    message: String,
    val requestedBackend: Accelerator,
    val fallbackBackend: Accelerator?,
    cause: Throwable? = null
) : RuntimeException(message, cause)