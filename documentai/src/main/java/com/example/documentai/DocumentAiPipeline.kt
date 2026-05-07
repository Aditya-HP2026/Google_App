package com.example.documentai

import android.content.Context
import android.graphics.Bitmap
import com.example.documentai.model.ClassificationResult
import com.example.documentai.model.OcrBox
import com.example.documentai.model.OcrResult
import com.example.documentai.model.PipelineResult
import com.example.documentai.model.TextLine

/**
 * Public entry-point for the on-device Document-AI pipeline.
 *
 * Wraps three ONNX models:
 *   1. `paddleocr/det.onnx`       – PP-OCRv5 text detector
 *   2. `paddleocr/rec.onnx`       – PP-OCRv5 text recognizer
 *   3. `distilbert/model.onnx`    – DistilBERT subject classifier
 *
 * **Initialization** loads and JIT-compiles all three models from the bundled
 * assets – this can take 1-3 seconds.  Call this constructor **off the main
 * thread** (e.g. inside a coroutine or an executor).
 *
 * **Thread safety** – each method is stateless between calls and safe to call
 * from a single background thread.  Do not call concurrently from multiple
 * threads without external synchronization.
 *
 * **Lifecycle** – call [close] when you are done to release native memory.
 *
 * Usage example:
 * ```kotlin
 * val pipeline = DocumentAiPipeline(context)
 *
 * // OCR only
 * val ocr: OcrResult = pipeline.extractText(bitmap)
 *
 * // Classification only (when you already have text)
 * val cls: ClassificationResult = pipeline.classifyText("F = ma")
 *
 * // Full pipeline: OCR → classify
 * val result: PipelineResult = pipeline.runPipeline(bitmap)
 * println("${result.ocr.text}  →  ${result.classification.label}")
 *
 * pipeline.close()
 * ```
 */
class DocumentAiPipeline(context: Context) : AutoCloseable {

    private val detector    = OcrDetector(context)
    private val postProcess = DbPostProcess()
    private val recognizer  = OcrRecognizer(context)
    private val ctcDecoder  = CtcDecoder(context)
    private val classifier  = DistilClassifier(context)

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Detects and recognizes all text regions in [bitmap].
     *
     * Pipeline:
     *   det.onnx  → DB post-process  → crop extraction  → rec.onnx  → CTC decode
     *
     * @return [OcrResult] with the full concatenated text and per-line details.
     */
    fun extractText(bitmap: Bitmap): OcrResult {
        // 1. Detect: run det.onnx + DB post-process
        val detOut = detector.detect(bitmap)
        val boxes  = postProcess.process(
            detOut.probMap, detOut.mapH, detOut.mapW,
            detOut.scaleH,  detOut.scaleW
        )
        if (boxes.isEmpty()) return OcrResult("", emptyList())

        // 2. Crop each detected region
        val crops = CropUtils.cropBoxes(bitmap, boxes)

        // 3. Recognise every crop → CTC decode
        val logitsList = recognizer.recognize(crops)

        // 4. Build result
        val lines = ArrayList<TextLine>(boxes.size)
        for (i in boxes.indices) {
            val text = ctcDecoder.decode(logitsList[i])
            if (text.isBlank()) continue
            val box = OcrBox(Array(4) { j ->
                floatArrayOf(boxes[i][j * 2], boxes[i][j * 2 + 1])
            })
            lines += TextLine(text, box)
        }
        return OcrResult(
            text  = lines.joinToString("\n") { it.text },
            lines = lines
        )
    }

    /**
     * Classifies [text] into one of: **biology**, **chemistry**, **maths**, **physics**.
     *
     * Pipeline:
     *   BERT WordPiece tokenize  → distilbert/model.onnx  → softmax → label
     *
     * @return [ClassificationResult] with label, confidence, and all class scores.
     */
    fun classifyText(text: String): ClassificationResult = classifier.classify(text)

    /**
     * Runs the full end-to-end pipeline on [bitmap]:
     *   extractText(bitmap)  →  classifyText(ocrResult.text)
     *
     * @return [PipelineResult] combining both [OcrResult] and [ClassificationResult].
     */
    fun runPipeline(bitmap: Bitmap): PipelineResult {
        val ocr = extractText(bitmap)
        val cls = classifier.classify(ocr.text.ifBlank { " " })
        return PipelineResult(ocr, cls)
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    override fun close() {
        detector.close()
        recognizer.close()
        classifier.close()
    }
}
