package com.example.googleapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.example.documentai.DocumentAiPipeline
import com.example.gemmaimage.GemmaImagePipeline
import com.example.gemmaimage.GemmaInitException
import com.example.semanticsearch.SemanticSearchEngine
import com.example.semanticsearch.model.SearchResult
import com.example.googleapp.ui.theme.*
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import java.io.File
import java.util.concurrent.Executors

private const val PERF_TAG = "PerfLog"
private const val MAX_GEMMA_BATCH_SIZE = 5

// ─── Data class for a classified item ────────────────────────────────────────

data class ClassifiedImage(
    val id: Long,                // Unique ID for stable keys
    val thumbnail: Bitmap,       // Small thumbnail for UI (max 400px)
    val label: String,
    val confidence: Float,
    val keywords: List<String>,
    val ocrText: String,
    val pipeline: String,
    val sourceName: String,      // e.g. "photo_1.jpg" or "document.pdf"
    val pageCount: Int = 1       // For PDFs: how many pages were classified
)

// ─── Subject metadata ─────────────────────────────────────────────────────────

data class SubjectMeta(
    val label: String,
    val displayName: String,
    val headerColor: Color,
    val circleColor: Color,
    val icon: ImageVector
)

val ALL_SUBJECT_META = listOf(
    SubjectMeta("maths",           "Math",            MathColor,          MathCircle,          Icons.Default.MenuBook),
    SubjectMeta("physics",         "Physics",         PhysicsColor,       PhysicsCircle,       Icons.Default.Adjust),
    SubjectMeta("chemistry",       "Chemistry",       ChemistryColor,     ChemistryCircle,     Icons.Default.Science),
    SubjectMeta("biology",         "Biology",         BiologyColor,       BiologyCircle,       Icons.Default.Eco),
    SubjectMeta("history",         "History",         HistoryColor,       HistoryCircle,       Icons.Default.HistoryEdu),
    SubjectMeta("geography",       "Geography",       GeographyColor,     GeographyCircle,     Icons.Default.Public),
    SubjectMeta("civics",          "Civics",          CivicsColor,        CivicsCircle,        Icons.Default.AccountBalance),
    SubjectMeta("science",         "Science",         ScienceColor,       ScienceCircle,       Icons.Default.Biotech),
    SubjectMeta("social_studies",  "Social Studies",  SocialStudiesColor, SocialStudiesCircle, Icons.Default.Groups),
    SubjectMeta("accounting",      "Accounting",      AccountingColor,    AccountingCircle,    Icons.Default.Calculate),
    SubjectMeta("business_studies","Business",        BusinessColor,      BusinessCircle,      Icons.Default.TrendingUp)
)

// ─── ID generator ─────────────────────────────────────────────────────────────
private var nextClassifiedId = 0L
private fun nextId(): Long = ++nextClassifiedId

private data class GemmaPipelineLoadResult(
    val pipeline: GemmaImagePipeline?,
    val message: String?
)

// ─── Activity ─────────────────────────────────────────────────────────────────

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            GoogleAppTheme {
                MainScreen()
            }
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

private const val THUMBNAIL_MAX_SIZE = 400
private const val MAX_PDF_PAGES = 10

fun Modifier.dashedBorder(
    color: Color,
    cornerRadius: Dp = 12.dp,
    strokeWidth: Dp = 1.5.dp,
    dashLength: Float = 12f,
    gapLength: Float = 8f
): Modifier = this.drawWithContent {
    drawContent()
    val pathEffect = PathEffect.dashPathEffect(floatArrayOf(dashLength, gapLength), 0f)
    drawRoundRect(
        color = color,
        style = Stroke(width = strokeWidth.toPx(), pathEffect = pathEffect),
        cornerRadius = CornerRadius(cornerRadius.toPx())
    )
}

/** Create a small thumbnail to avoid holding full-res bitmaps in memory / compose state */
private fun createThumbnail(bitmap: Bitmap): Bitmap {
    val maxDim = maxOf(bitmap.width, bitmap.height)
    if (maxDim <= THUMBNAIL_MAX_SIZE) return bitmap
    val scale = THUMBNAIL_MAX_SIZE.toFloat() / maxDim
    val w = (bitmap.width * scale).toInt().coerceAtLeast(1)
    val h = (bitmap.height * scale).toInt().coerceAtLeast(1)
    return Bitmap.createScaledBitmap(bitmap, w, h, true)
}

private fun loadBitmapFromUri(context: android.content.Context, uri: Uri): Bitmap? = try {
    context.contentResolver.openInputStream(uri)?.use { stream ->
        // Decode with downsampling for large images
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeStream(stream, null, options)
        stream.close()

        val maxDim = maxOf(options.outWidth, options.outHeight)
        val sampleSize = (maxDim / 1024).coerceAtLeast(1)

        context.contentResolver.openInputStream(uri)?.use { stream2 ->
            val decodeOptions = BitmapFactory.Options().apply { inSampleSize = sampleSize }
            BitmapFactory.decodeStream(stream2, null, decodeOptions)
        }
    }
} catch (_: Exception) { null }

/** Render first page of PDF as thumbnail, return all pages (max 10) for classification */
private fun renderPdfPages(context: android.content.Context, uri: Uri, maxPages: Int = MAX_PDF_PAGES): List<Bitmap> {
    val bitmaps = mutableListOf<Bitmap>()
    try {
        val fd = context.contentResolver.openFileDescriptor(uri, "r") ?: return bitmaps
        fd.use { descriptor ->
            PdfRenderer(descriptor).use { pdf ->
                val pageCount = minOf(pdf.pageCount, maxPages)
                for (i in 0 until pageCount) {
                    pdf.openPage(i).use { page ->
                        // Render at 1x scale (not 2x) to reduce memory
                        val w = page.width; val h = page.height
                        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                        bmp.eraseColor(android.graphics.Color.WHITE)
                        page.render(bmp, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                        bitmaps += bmp
                    }
                }
            }
        }
    } catch (_: Exception) {}
    return bitmaps
}

private fun getPdfPageCount(context: android.content.Context, uri: Uri): Int = try {
    val fd = context.contentResolver.openFileDescriptor(uri, "r")
    fd?.use { PdfRenderer(it).use { pdf -> pdf.pageCount } } ?: 0
} catch (_: Exception) { 0 }

private fun isPdf(context: android.content.Context, uri: Uri): Boolean =
    (context.contentResolver.getType(uri) ?: "") == "application/pdf"

private fun getFileName(context: android.content.Context, uri: Uri): String {
    var name = "file"
    try {
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && cursor.moveToFirst()) {
                name = cursor.getString(idx) ?: "file"
            }
        }
    } catch (_: Exception) {}
    return name
}

// ─── Main Screen with bottom tabs ────────────────────────────────────────────

@Composable
fun MainScreen() {
    val context = LocalContext.current
    var selectedTab by remember { mutableIntStateOf(0) }

    var searchEngine by remember { mutableStateOf<SemanticSearchEngine?>(null) }
    var searchEngineError by remember { mutableStateOf("") }

    // Shared classified images state (accessible by both tabs)
    var classifiedImages by remember { mutableStateOf(listOf<ClassifiedImage>()) }

    // Log recomposition of MainScreen
    SideEffect {
        Log.d(PERF_TAG, "MainScreen recomposed, tab=$selectedTab, images=${classifiedImages.size}")
    }

    LaunchedEffect(Unit) {
        val initResult = withContext(Dispatchers.IO) {
            val dbFile = File(context.getExternalFilesDir(null), "knowledge.db")
            runCatching { SemanticSearchEngine(context, dbFile) }
        }
        initResult
            .onSuccess {
                searchEngine = it
                searchEngineError = ""
            }
            .onFailure {
                searchEngineError = it.message ?: "Failed to initialize search engine"
            }
    }

    Scaffold(
        bottomBar = {
            NavigationBar(containerColor = Color.White) {
                NavigationBarItem(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    icon = { Icon(Icons.Default.PhotoLibrary, contentDescription = "Classify") },
                    label = { Text("Classify") }
                )
                NavigationBarItem(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    icon = { Icon(Icons.Default.Search, contentDescription = "Search") },
                    label = { Text("Search") }
                )
            }
        }
    ) { innerPadding ->
        Box(modifier = Modifier.padding(innerPadding)) {
            when (selectedTab) {
                0 -> ClassifyTab(
                    searchEngine = searchEngine,
                    classifiedImages = classifiedImages,
                    onImagesUpdated = { classifiedImages = it }
                )
                1 -> SearchTab(
                    searchEngine = searchEngine,
                    engineError = searchEngineError,
                    classifiedImages = classifiedImages,
                    onClearImages = { classifiedImages = emptyList() }
                )
            }
        }
    }
}

// ─── CLASSIFY TAB ─────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ClassifyTab(
    searchEngine: SemanticSearchEngine?,
    classifiedImages: List<ClassifiedImage>,
    onImagesUpdated: (List<ClassifiedImage>) -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var isProcessing by remember { mutableStateOf(false) }
    var processingProgress by remember { mutableStateOf("") }
    var processedCount by remember { mutableIntStateOf(0) }
    var totalToProcess by remember { mutableIntStateOf(0) }
    var selectedPipeline by remember { mutableIntStateOf(0) }
    var distilPipeline by remember { mutableStateOf<DocumentAiPipeline?>(null) }
    var gemmaPipeline by remember { mutableStateOf<GemmaImagePipeline?>(null) }
    var processingJob by remember { mutableStateOf<Job?>(null) }

    // Detail dialog state
    var selectedSubject by remember { mutableStateOf<SubjectMeta?>(null) }

    val gemmaDispatcher = remember {
        Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "GemmaInference").apply {
                priority = Thread.MIN_PRIORITY
            }
        }.asCoroutineDispatcher()
    }

    DisposableEffect(Unit) {
        onDispose {
            processingJob?.cancel()
            gemmaDispatcher.close()
        }
    }

    // File picker that supports images AND documents
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenMultipleDocuments()
    ) { uris: List<Uri> ->
        if (uris.isEmpty()) return@rememberLauncherForActivityResult

        val pipelineChoice = selectedPipeline
        val queuedUris = if (pipelineChoice == 1 && uris.size > MAX_GEMMA_BATCH_SIZE) {
            Toast.makeText(
                context,
                "Gemma will process the first $MAX_GEMMA_BATCH_SIZE files to keep the app responsive.",
                Toast.LENGTH_LONG
            ).show()
            uris.take(MAX_GEMMA_BATCH_SIZE)
        } else {
            uris
        }

        processingJob?.cancel()
        isProcessing = true
        processingProgress = "Preparing…"
        processedCount = 0
        totalToProcess = queuedUris.size

        val existingImages = classifiedImages
        processingJob = scope.launch {
            val batchStartMs = System.currentTimeMillis()
            val results = mutableListOf<ClassifiedImage>()
            Log.d(PERF_TAG, "── Processing batch start: ${queuedUris.size} file(s) ──")

            try {
                var localDistil = distilPipeline
                var localGemma = gemmaPipeline

                if (localDistil == null) {
                    processingProgress = if (pipelineChoice == 0) "Loading OCR + Classifier…" else "Loading OCR…"
                    localDistil = withContext(Dispatchers.Default) { DocumentAiPipeline(context) }
                    distilPipeline = localDistil
                }

                if (pipelineChoice == 1) {
                    processingProgress = "Loading Gemma…"
                    val loadResult = withContext(gemmaDispatcher) {
                        loadGemmaPipeline(context, localGemma)
                    }
                    localGemma = loadResult.pipeline
                    if (loadResult.message != null) {
                        Toast.makeText(context, loadResult.message, Toast.LENGTH_LONG).show()
                    }
                    if (localGemma == null) {
                        isProcessing = false
                        processingProgress = ""
                        totalToProcess = 0
                        processingJob = null
                        return@launch
                    }
                    gemmaPipeline = localGemma
                }

                for ((idx, uri) in queuedUris.withIndex()) {
                    yield()
                    val fileName = withContext(Dispatchers.IO) { getFileName(context, uri) }
                    processingProgress = "Classifying ${idx + 1}/${queuedUris.size}: $fileName"

                    val dispatcher = if (pipelineChoice == 1) gemmaDispatcher else Dispatchers.Default
                    val item = withContext(dispatcher) {
                        classifyUri(
                            context = context,
                            uri = uri,
                            fileName = fileName,
                            selectedPipeline = pipelineChoice,
                            distilPipeline = localDistil,
                            gemmaPipeline = localGemma
                        )
                    }

                    if (item != null) {
                        results += item
                        onImagesUpdated(existingImages + results.toList())
                    }
                    processedCount = idx + 1
                    delay(80)
                }

                Log.d(PERF_TAG, "── Batch complete: ${System.currentTimeMillis() - batchStartMs}ms total, ${results.size} results ──")

                isProcessing = false
                processingProgress = ""
                processingJob = null

                if (results.isEmpty()) {
                    Toast.makeText(context, "No content detected", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(context, "Classified ${results.size} item(s)", Toast.LENGTH_SHORT).show()
                }

                if (searchEngine != null && results.isNotEmpty()) {
                    scope.launch(Dispatchers.IO) {
                        results.forEach { img ->
                            try {
                                searchEngine.indexDocument(
                                    sourcePath = img.sourceName,
                                    subject = img.label,
                                    keywords = img.keywords.ifEmpty { img.ocrText.split(" ").filter { it.isNotBlank() }.take(10) },
                                    textChunk = img.ocrText.ifEmpty { img.keywords.joinToString(" ") }
                                )
                            } catch (e: Exception) {
                                Log.e("SearchIndex", "Failed to index ${img.sourceName}", e)
                            }
                        }
                    }
                }
            } catch (_: CancellationException) {
                isProcessing = false
                processingProgress = ""
                processedCount = 0
                totalToProcess = 0
                processingJob = null
                Toast.makeText(context, "Processing cancelled", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Log.e(PERF_TAG, "Batch processing failed", e)
                isProcessing = false
                processingProgress = ""
                processedCount = 0
                totalToProcess = 0
                processingJob = null
                Toast.makeText(context, "Processing failed: ${e.message?.take(80)}", Toast.LENGTH_LONG).show()
            }
        }
    }

    // ── UI ────────────────────────────────────────────────────────────────────
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(AppBackground)
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
        // Header
        item(key = "header") {
            Spacer(Modifier.height(48.dp))
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                Text("✨", fontSize = 22.sp)
                Spacer(Modifier.width(6.dp))
                Text("AI Subject Organizer", fontSize = 22.sp, fontWeight = FontWeight.Bold, color = AppTitleColor)
            }
            Spacer(Modifier.height(6.dp))
            Text(
                "Upload images or PDFs — AI automatically sorts by subject",
                fontSize = 13.sp, color = AppAccent, textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 24.dp)
            )
            Spacer(Modifier.height(16.dp))
        }

        // Pipeline toggle
        item(key = "pipeline") {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                elevation = CardDefaults.cardElevation(2.dp)
            ) {
                Column(Modifier.padding(12.dp)) {
                    Text("Pipeline", fontWeight = FontWeight.SemiBold, fontSize = 13.sp, color = AppTitleColor)
                    Spacer(Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        PipelineChip("PaddleOCR + DistilBERT", selectedPipeline == 0) { selectedPipeline = 0 }
                        PipelineChip("Gemma 4 E2B", selectedPipeline == 1) { selectedPipeline = 1 }
                    }
                    if (selectedPipeline == 1) {
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "⚠ Requires gemma-4-E2B-it.litertlm (2.58 GB). Prefers GPU, falls back to CPU.",
                            fontSize = 11.sp, color = Color(0xFF888888)
                        )
                    }
                }
            }
            Spacer(Modifier.height(16.dp))
        }

        // Upload card
        item(key = "upload") {
            UploadCard(
                onChooseFiles = {
                    filePickerLauncher.launch(arrayOf("image/*", "application/pdf"))
                },
                isProcessing = isProcessing,
                processingProgress = processingProgress
            )
            Spacer(Modifier.height(20.dp))
        }

        // Subject grid (2 columns) - use stable keys
        val rows = ALL_SUBJECT_META.chunked(2)
        items(rows, key = { row -> row.map { it.label }.joinToString() }) { row ->
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                for (meta in row) {
                    val count = remember(classifiedImages) {
                        classifiedImages.count { it.label == meta.label }
                    }
                    SubjectCard(
                        meta = meta,
                        count = count,
                        firstImage = remember(classifiedImages) {
                            classifiedImages.firstOrNull { it.label == meta.label }
                        },
                        modifier = Modifier.weight(1f),
                        onClick = { selectedSubject = meta }
                    )
                }
                if (row.size == 1) Spacer(Modifier.weight(1f))
            }
            Spacer(Modifier.height(12.dp))
        }

            item(key = "bottom_spacer") { Spacer(Modifier.height(24.dp)) }
        }

        if (isProcessing) {
            ProcessingOverlay(
                progressText = processingProgress,
                completed = processedCount,
                total = totalToProcess,
                onCancel = { processingJob?.cancel() }
            )
        }
    }

    // ── Detail dialog when subject tile is tapped ─────────────────────────────
    selectedSubject?.let { meta ->
        val subjectImages = remember(classifiedImages, meta) {
            classifiedImages.filter { it.label == meta.label }
        }
        SubjectDetailDialog(
            meta = meta,
            images = subjectImages,
            onDismiss = { selectedSubject = null },
            onDeleteImage = { img ->
                onImagesUpdated(classifiedImages.filter { it.id != img.id })
                if (searchEngine != null) {
                    scope.launch(Dispatchers.IO) {
                        runCatching { searchEngine.removeDocument(img.sourceName) }
                    }
                }
            }
        )
    }
}

private fun loadGemmaPipeline(
    context: android.content.Context,
    existing: GemmaImagePipeline?
): GemmaPipelineLoadResult {
    if (existing != null) return GemmaPipelineLoadResult(existing, null)

    val modelFile = File(context.getExternalFilesDir(null), GemmaImagePipeline.MODEL_FILE_NAME)
    if (!modelFile.exists()) {
        return GemmaPipelineLoadResult(null, "Place ${GemmaImagePipeline.MODEL_FILE_NAME} in app files")
    }

    return try {
        val pipe = GemmaImagePipeline(context)
        val backend = pipe.activeBackend?.displayName ?: "unknown"
        val initMs = pipe.initResult?.initTimeMs ?: 0
        val didFallback = pipe.initResult?.didFallback == true
        val msg = if (didFallback) "⚠ GPU unavailable — using CPU (${initMs}ms)"
        else "✅ Running on $backend (${initMs}ms)"
        GemmaPipelineLoadResult(pipe, msg)
    } catch (e: Exception) {
        Log.e("GemmaPipeline", "Failed to load Gemma model", e)
        val friendlyMsg = if (e is GemmaInitException) "All backends failed" else "Load error: ${e.message?.take(80)}"
        GemmaPipelineLoadResult(null, friendlyMsg)
    }
}

private fun classifyUri(
    context: android.content.Context,
    uri: Uri,
    fileName: String,
    selectedPipeline: Int,
    distilPipeline: DocumentAiPipeline?,
    gemmaPipeline: GemmaImagePipeline?
): ClassifiedImage? {
    val fileStartMs = System.currentTimeMillis()
    Log.d(PERF_TAG, "Start file: $fileName")

    if (isPdf(context, uri)) {
        val totalPages = getPdfPageCount(context, uri)
        val renderStartMs = System.currentTimeMillis()
        val pages = renderPdfPages(context, uri, MAX_PDF_PAGES)
        Log.d(PERF_TAG, "PDF render ${pages.size} pages: ${System.currentTimeMillis() - renderStartMs}ms")
        if (pages.isEmpty()) return null

        val thumb = createThumbnail(pages.first())
        val item = try {
            if (selectedPipeline == 0) {
                val res = distilPipeline?.runPipeline(pages.first()) ?: return null
                ClassifiedImage(
                    id = nextId(),
                    thumbnail = thumb,
                    label = res.classification.label,
                    confidence = res.classification.confidence,
                    keywords = emptyList(),
                    ocrText = res.ocr.text,
                    pipeline = "PaddleOCR+DistilBERT",
                    sourceName = fileName,
                    pageCount = totalPages
                )
            } else {
                val pipe = gemmaPipeline ?: return null
                val backendLabel = pipe.activeBackend?.displayName ?: "LiteRT"
                
                // ── TIMING STAGE 1: OCR extraction ──
                val ocrStartMs = System.currentTimeMillis()
                val ocrText = distilPipeline?.extractText(pages.first())?.text.orEmpty().trim()
                val ocrTimeMs = System.currentTimeMillis() - ocrStartMs
                Log.d(PERF_TAG, "┌─── CLASSIFICATION PIPELINE (PDF, page 1) ───")
                Log.d(PERF_TAG, "│ STAGE 1 (OCR):        $ocrTimeMs ms | chars=${ocrText.length}")
                
                // ── TIMING STAGE 2: Classification (Text vs Vision) ──
                val classifyStartMs = System.currentTimeMillis()
                val isTextMode = ocrText.length >= 12  // OPTIMIZATION: lowered from 24 → 12
                val res = if (isTextMode) {
                    Log.d(PERF_TAG, "│ STAGE 2 (Classify):   TEXT mode (OCR strong)")
                    pipe.classifyText(ocrText)
                } else {
                    Log.d(PERF_TAG, "│ STAGE 2 (Classify):   VISION mode (OCR weak/missing)")
                    pipe.classify(pages.first())
                }
                val classifyTimeMs = System.currentTimeMillis() - classifyStartMs
                Log.d(PERF_TAG, "│                       $classifyTimeMs ms")
                Log.d(PERF_TAG, "└─────────────────────────────────────────────")
                
                val totalMs = System.currentTimeMillis() - fileStartMs
                Log.d(PERF_TAG, "📊 PDF FILE TOTAL: $totalMs ms | OCR: $ocrTimeMs ms | Classify: $classifyTimeMs ms")
                
                ClassifiedImage(
                    id = nextId(),
                    thumbnail = thumb,
                    label = res.label,
                    confidence = res.confidence,
                    keywords = res.keywords,
                    ocrText = if (ocrText.isNotBlank()) ocrText else res.evidence,
                    pipeline = "Gemma 4 E2B ${if (isTextMode) "Text" else "Vision"} ($backendLabel)",
                    sourceName = fileName,
                    pageCount = totalPages
                )
            }
        } finally {
            pages.forEach { if (it != thumb) it.recycle() }
        }
        Log.d(PERF_TAG, "File total: ${System.currentTimeMillis() - fileStartMs}ms")
        return item
    }

    val loadStartMs = System.currentTimeMillis()
    val bitmap = loadBitmapFromUri(context, uri) ?: return null
    Log.d(PERF_TAG, "Bitmap load: ${System.currentTimeMillis() - loadStartMs}ms, size=${bitmap.width}x${bitmap.height}")
    val thumb = createThumbnail(bitmap)

    return try {
        val item = if (selectedPipeline == 0) {
            val res = distilPipeline?.runPipeline(bitmap) ?: return null
            ClassifiedImage(
                id = nextId(),
                thumbnail = thumb,
                label = res.classification.label,
                confidence = res.classification.confidence,
                keywords = emptyList(),
                ocrText = res.ocr.text,
                pipeline = "PaddleOCR+DistilBERT",
                sourceName = fileName
            )
        } else {
            val pipe = gemmaPipeline ?: return null
            val backendLabel = pipe.activeBackend?.displayName ?: "LiteRT"
            
            // ── TIMING STAGE 1: OCR extraction ──
            val ocrStartMs = System.currentTimeMillis()
            val ocrText = distilPipeline?.extractText(bitmap)?.text.orEmpty().trim()
            val ocrTimeMs = System.currentTimeMillis() - ocrStartMs
            Log.d(PERF_TAG, "┌─── CLASSIFICATION PIPELINE (IMAGE) ───────────")
            Log.d(PERF_TAG, "│ STAGE 1 (OCR):        $ocrTimeMs ms | chars=${ocrText.length}")
            
            // ── TIMING STAGE 2: Classification (Text vs Vision) ──
            val classifyStartMs = System.currentTimeMillis()
            val isTextMode = ocrText.length >= 12  // OPTIMIZATION: lowered from 24 → 12
            val res = if (isTextMode) {
                Log.d(PERF_TAG, "│ STAGE 2 (Classify):   TEXT mode (OCR strong)")
                pipe.classifyText(ocrText)
            } else {
                Log.d(PERF_TAG, "│ STAGE 2 (Classify):   VISION mode (OCR weak/missing)")
                pipe.classify(bitmap)
            }
            val classifyTimeMs = System.currentTimeMillis() - classifyStartMs
            Log.d(PERF_TAG, "│                       $classifyTimeMs ms")
            Log.d(PERF_TAG, "└─────────────────────────────────────────────")
            
            val totalMs = System.currentTimeMillis() - fileStartMs
            Log.d(PERF_TAG, "📊 IMAGE FILE TOTAL: $totalMs ms | OCR: $ocrTimeMs ms | Classify: $classifyTimeMs ms")
            
            ClassifiedImage(
                id = nextId(),
                thumbnail = thumb,
                label = res.label,
                confidence = res.confidence,
                keywords = res.keywords,
                ocrText = if (ocrText.isNotBlank()) ocrText else res.evidence,
                pipeline = "Gemma 4 E2B ${if (isTextMode) "Text" else "Vision"} ($backendLabel)",
                sourceName = fileName
            )
        }
        Log.d(PERF_TAG, "File total: ${System.currentTimeMillis() - fileStartMs}ms")
        item
    } finally {
        if (bitmap != thumb) bitmap.recycle()
    }
}

@Composable
fun PipelineChip(label: String, selected: Boolean, onClick: () -> Unit) {
    FilterChip(
        selected = selected,
        onClick = onClick,
        label = { Text(label, fontSize = 12.sp) },
        colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = AppAccent,
            selectedLabelColor = Color.White
        )
    )
}

// ─── Upload Card ──────────────────────────────────────────────────────────────

@Composable
fun UploadCard(onChooseFiles: () -> Unit, isProcessing: Boolean, processingProgress: String) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Color.White)
            .dashedBorder(color = DashedBorderColor, cornerRadius = 12.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(28.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                modifier = Modifier.size(64.dp).background(UploadCircleBg, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.FileUpload, contentDescription = "Upload", tint = AppAccent, modifier = Modifier.size(30.dp))
            }
            Spacer(Modifier.height(14.dp))
            Text("Upload Your Files", fontSize = 17.sp, fontWeight = FontWeight.Bold, color = AppTitleColor)
            Spacer(Modifier.height(4.dp))
            Text("AI will automatically detect and organize by subject", fontSize = 13.sp, color = AppAccent, textAlign = TextAlign.Center)
            Spacer(Modifier.height(18.dp))
            if (isProcessing) {
                CircularProgressIndicator(modifier = Modifier.size(36.dp), color = AppAccent, strokeWidth = 3.dp)
                Spacer(Modifier.height(10.dp))
                Text(processingProgress, fontSize = 12.sp, color = AppAccent, fontWeight = FontWeight.Medium)
            } else {
                Button(
                    onClick = onChooseFiles,
                    colors = ButtonDefaults.buttonColors(containerColor = ButtonDark),
                    shape = RoundedCornerShape(8.dp),
                    contentPadding = PaddingValues(horizontal = 22.dp, vertical = 12.dp)
                ) {
                    Icon(Icons.Default.FileUpload, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Choose Files", fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
                }
            }
            Spacer(Modifier.height(10.dp))
            Text("Supports JPG, PNG, PDF", fontSize = 11.sp, color = Color(0xFFAAAAAA))
        }
    }
}

@Composable
fun ProcessingOverlay(
    progressText: String,
    completed: Int,
    total: Int,
    onCancel: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.28f)),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp),
            shape = RoundedCornerShape(18.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                CircularProgressIndicator(color = AppAccent, strokeWidth = 3.dp)
                Spacer(Modifier.height(16.dp))
                Text(
                    if (progressText.isBlank()) "Processing…" else progressText,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = AppTitleColor,
                    textAlign = TextAlign.Center
                )
                if (total > 0) {
                    Spacer(Modifier.height(12.dp))
                    LinearProgressIndicator(
                        progress = { completed.toFloat() / total.coerceAtLeast(1) },
                        modifier = Modifier.fillMaxWidth(),
                        color = AppAccent,
                        trackColor = AppBackground
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "$completed of $total completed",
                        fontSize = 12.sp,
                        color = EmptyStateText
                    )
                }
                Spacer(Modifier.height(16.dp))
                OutlinedButton(onClick = onCancel) {
                    Icon(Icons.Default.Close, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Cancel")
                }
            }
        }
    }
}

// ─── Subject Card (clickable tile) ────────────────────────────────────────────

@Composable
fun SubjectCard(
    meta: SubjectMeta,
    count: Int,
    firstImage: ClassifiedImage?,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Card(
        modifier = modifier.clickable(enabled = count > 0) { onClick() },
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth().background(meta.headerColor).padding(horizontal = 12.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(meta.icon, contentDescription = meta.displayName, tint = Color.White, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Column {
                    Text(meta.displayName, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp, lineHeight = 16.sp)
                    Text("$count items", color = Color.White.copy(alpha = 0.85f), fontSize = 11.sp)
                }
            }
            if (firstImage == null) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Box(modifier = Modifier.size(40.dp).background(meta.circleColor, CircleShape))
                    Spacer(Modifier.height(8.dp))
                    Text("No items yet", color = EmptyStateText, fontSize = 11.sp, fontWeight = FontWeight.Medium)
                }
            } else {
                // Show compact summary — just first item thumbnail + count
                Column(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
                    Image(
                        bitmap = firstImage.thumbnail.asImageBitmap(),
                        contentDescription = null,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(80.dp)
                            .clip(RoundedCornerShape(6.dp)),
                        contentScale = ContentScale.Crop
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        "${(firstImage.confidence * 100).toInt()}% • ${firstImage.pipeline}",
                        fontSize = 9.sp, fontWeight = FontWeight.Medium, color = meta.headerColor,
                        maxLines = 1, overflow = TextOverflow.Ellipsis
                    )
                    if (count > 1) {
                        Text(
                            "Tap to view all $count items",
                            fontSize = 9.sp, color = AppAccent, fontWeight = FontWeight.Medium
                        )
                    } else {
                        Text(
                            firstImage.sourceName,
                            fontSize = 9.sp, color = EmptyStateText, maxLines = 1, overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
        }
    }
}

// ─── Subject Detail Dialog (full list of classified items) ────────────────────

@Composable
fun SubjectDetailDialog(
    meta: SubjectMeta,
    images: List<ClassifiedImage>,
    onDismiss: () -> Unit,
    onDeleteImage: (ClassifiedImage) -> Unit = {}
) {
    // Auto-dismiss when all items have been deleted
    LaunchedEffect(images.isEmpty()) {
        if (images.isEmpty()) onDismiss()
    }

    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.85f),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column {
                // Header
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(meta.headerColor)
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(meta.icon, contentDescription = null, tint = Color.White, modifier = Modifier.size(24.dp))
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text(meta.displayName, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 18.sp)
                        Text("${images.size} classified items", color = Color.White.copy(0.85f), fontSize = 12.sp)
                    }
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Default.Close, contentDescription = "Close", tint = Color.White)
                    }
                }

                // Items list
                LazyColumn(
                    modifier = Modifier.fillMaxSize().padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(images, key = { it.id }) { img ->
                        var showDeleteConfirm by remember { mutableStateOf(false) }

                        Card(
                            shape = RoundedCornerShape(10.dp),
                            colors = CardDefaults.cardColors(containerColor = meta.circleColor.copy(alpha = 0.15f)),
                            elevation = CardDefaults.cardElevation(0.dp)
                        ) {
                            Column(Modifier.padding(10.dp)) {
                                Image(
                                    bitmap = img.thumbnail.asImageBitmap(),
                                    contentDescription = "Classified as ${meta.displayName}",
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .heightIn(min = 100.dp, max = 200.dp)
                                        .clip(RoundedCornerShape(8.dp)),
                                    contentScale = ContentScale.Fit
                                )
                                Spacer(Modifier.height(8.dp))

                                // File info row with delete button
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Icon(Icons.Default.Description, contentDescription = null, modifier = Modifier.size(14.dp), tint = meta.headerColor)
                                    Spacer(Modifier.width(4.dp))
                                    Text(
                                        img.sourceName,
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.Medium,
                                        color = AppTitleColor,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis,
                                        modifier = Modifier.weight(1f)
                                    )
                                    if (img.pageCount > 1) {
                                        Text(" (${img.pageCount}p)", fontSize = 11.sp, color = EmptyStateText)
                                    }
                                    // Delete button
                                    IconButton(
                                        onClick = { showDeleteConfirm = true },
                                        modifier = Modifier.size(28.dp)
                                    ) {
                                        Icon(
                                            Icons.Default.Delete,
                                            contentDescription = "Delete",
                                            tint = Color(0xFFE53935),
                                            modifier = Modifier.size(18.dp)
                                        )
                                    }
                                }

                                Spacer(Modifier.height(4.dp))

                                // Confidence + pipeline
                                Text(
                                    "${(img.confidence * 100).toInt()}% confidence • ${img.pipeline}",
                                    fontSize = 11.sp, color = meta.headerColor, fontWeight = FontWeight.Medium
                                )

                                // Keywords
                                if (img.keywords.isNotEmpty()) {
                                    Spacer(Modifier.height(6.dp))
                                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.fillMaxWidth()) {
                                        img.keywords.take(5).forEach { kw ->
                                            Box(
                                                modifier = Modifier
                                                    .background(meta.circleColor, RoundedCornerShape(4.dp))
                                                    .padding(horizontal = 6.dp, vertical = 2.dp)
                                            ) {
                                                Text(kw, fontSize = 9.sp, color = meta.headerColor)
                                            }
                                        }
                                    }
                                }

                                // OCR text preview
                                if (img.ocrText.isNotBlank()) {
                                    Spacer(Modifier.height(6.dp))
                                    Text(
                                        img.ocrText.take(120) + if (img.ocrText.length > 120) "…" else "",
                                        fontSize = 11.sp, color = Color(0xFF555555), lineHeight = 15.sp, maxLines = 3
                                    )
                                }
                            }
                        }

                        // Confirm delete dialog
                        if (showDeleteConfirm) {
                            AlertDialog(
                                onDismissRequest = { showDeleteConfirm = false },
                                title = { Text("Delete Image?") },
                                text = { Text("\"${img.sourceName}\" will be removed from the library and the search index.") },
                                confirmButton = {
                                    TextButton(onClick = {
                                        showDeleteConfirm = false
                                        onDeleteImage(img)
                                    }) {
                                        Text("Delete", color = Color(0xFFE53935))
                                    }
                                },
                                dismissButton = {
                                    TextButton(onClick = { showDeleteConfirm = false }) {
                                        Text("Cancel")
                                    }
                                }
                            )
                        }
                    }

                    item(key = "detail_bottom") { Spacer(Modifier.height(8.dp)) }
                }
            }
        }
    }
}

// ─── SEARCH TAB ───────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SearchTab(
    searchEngine: SemanticSearchEngine?,
    engineError: String,
    classifiedImages: List<ClassifiedImage>,
    onClearImages: () -> Unit = {}
) {
    val scope = rememberCoroutineScope()

    var query by remember { mutableStateOf("") }
    var results by remember { mutableStateOf(listOf<SearchResult>()) }
    var isSearching by remember { mutableStateOf(false) }
    var errorMsg by remember { mutableStateOf("") }
    var documentCount by remember { mutableIntStateOf(0) }
    var showClearConfirm by remember { mutableStateOf(false) }

    // Log recomposition
    SideEffect {
        Log.d(PERF_TAG, "SearchTab recomposed, results=${results.size}, images=${classifiedImages.size}")
    }

    LaunchedEffect(searchEngine) {
        documentCount = if (searchEngine != null) {
            withContext(Dispatchers.IO) { searchEngine.documentCount() }
        } else {
            0
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppBackground)
            .padding(horizontal = 16.dp)
    ) {
        Spacer(Modifier.height(48.dp))

        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("🔍", fontSize = 22.sp)
            Spacer(Modifier.width(6.dp))
            Text("Semantic Search", fontSize = 22.sp, fontWeight = FontWeight.Bold, color = AppTitleColor)
            Spacer(Modifier.weight(1f))
            // Clear database button — only shown when there are indexed documents
            if (searchEngine != null && documentCount > 0) {
                IconButton(onClick = { showClearConfirm = true }) {
                    Icon(
                        Icons.Default.Delete,
                        contentDescription = "Clear database",
                        tint = Color(0xFFE53935),
                        modifier = Modifier.size(22.dp)
                    )
                }
            }
        }
        Spacer(Modifier.height(6.dp))
        Text(
            "Search your indexed documents using natural language",
            fontSize = 13.sp, color = AppAccent, textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 8.dp)
        )

        if (searchEngine != null && documentCount > 0) {
            Spacer(Modifier.height(4.dp))
            Text("$documentCount documents indexed", fontSize = 11.sp, color = Color(0xFF4CAF50), fontWeight = FontWeight.Medium)
        }

        Spacer(Modifier.height(16.dp))

        if (engineError.isNotEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF3E0)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(Modifier.padding(16.dp)) {
                    Text("⚠ knowledge.db not found", fontWeight = FontWeight.Bold, fontSize = 14.sp, color = Color(0xFFE65100))
                    Spacer(Modifier.height(4.dp))
                    Text(
                        "Push knowledge.db via ADB:\nadb push knowledge.db /sdcard/Android/data/com.example.googleapp/files/",
                        fontSize = 12.sp, color = Color(0xFF795548)
                    )
                }
            }
            Spacer(Modifier.height(12.dp))
        }

        // Search bar
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                placeholder = { Text("e.g. chemical bonding ionic covalent") },
                modifier = Modifier.weight(1f),
                singleLine = true,
                shape = RoundedCornerShape(12.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = AppAccent,
                    unfocusedBorderColor = DashedBorderColor,
                    unfocusedContainerColor = Color.White,
                    focusedContainerColor = Color.White
                )
            )
            Button(
                onClick = {
                    if (query.isBlank() || searchEngine == null) return@Button
                    isSearching = true; errorMsg = ""
                    scope.launch {
                        try {
                            val res = withContext(Dispatchers.IO) { searchEngine.search(query.trim(), topK = 8) }
                            results = res
                            try { documentCount = searchEngine.documentCount() } catch (_: Exception) {}
                            if (res.isEmpty()) errorMsg = "No matching documents found."
                        } catch (e: Exception) {
                            Log.e("SearchTab", "Search crashed", e)
                            errorMsg = "Search error: ${e.message?.take(100)}"
                        }
                        isSearching = false
                    }
                },
                enabled = !isSearching && searchEngine != null && query.isNotBlank(),
                colors = ButtonDefaults.buttonColors(containerColor = AppAccent),
                shape = RoundedCornerShape(12.dp),
                contentPadding = PaddingValues(horizontal = 18.dp, vertical = 16.dp)
            ) {
                if (isSearching) CircularProgressIndicator(modifier = Modifier.size(18.dp), color = Color.White, strokeWidth = 2.dp)
                else Icon(Icons.Default.Search, contentDescription = "Search", modifier = Modifier.size(18.dp))
            }
        }

        Spacer(Modifier.height(12.dp))

        if (errorMsg.isNotEmpty()) {
            Text(errorMsg, fontSize = 13.sp, color = Color(0xFFE65100), modifier = Modifier.padding(vertical = 4.dp))
        }

        // Image viewer state
        var selectedImage by remember { mutableStateOf<Bitmap?>(null) }

        LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            items(results.size, key = { idx -> "search_result_$idx" }) { idx ->
                val result = results[idx]
                // Find matching thumbnail from classified images (in current session only)
                val matchingImage = remember(classifiedImages, result) {
                    classifiedImages.firstOrNull { it.sourceName == result.sourcePath }
                }
                SearchResultCard(
                    result = result,
                    thumbnail = matchingImage?.thumbnail,
                    onClick = { if (matchingImage?.thumbnail != null) selectedImage = matchingImage.thumbnail }
                )
            }
            item(key = "search_bottom") { Spacer(Modifier.height(24.dp)) }
        }

        // Show full-screen image viewer when a result is tapped
        selectedImage?.let { bmp ->
            ImageViewerDialog(
                image = bmp,
                onDismiss = { selectedImage = null }
            )
        }

        // Confirm clear database dialog
        if (showClearConfirm) {
            AlertDialog(
                onDismissRequest = { showClearConfirm = false },
                title = { Text("Clear Database?") },
                text = { Text("All $documentCount indexed documents will be permanently deleted from the search index. This cannot be undone.") },
                confirmButton = {
                    TextButton(onClick = {
                        showClearConfirm = false
                        scope.launch(Dispatchers.IO) {
                            runCatching { searchEngine?.clearAllDocuments() }
                            withContext(Dispatchers.Main) {
                                results = emptyList()
                                documentCount = 0
                                errorMsg = ""
                                onClearImages()
                            }
                        }
                    }) {
                        Text("Clear All", color = Color(0xFFE53935), fontWeight = FontWeight.Bold)
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showClearConfirm = false }) {
                        Text("Cancel")
                    }
                }
            )
        }
    }
}

@Composable
fun SearchResultCard(result: SearchResult, thumbnail: Bitmap? = null, onClick: () -> Unit = {}) {
    val meta = ALL_SUBJECT_META.firstOrNull { it.label == result.subject }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .then(if (thumbnail != null) Modifier.clickable { onClick() } else Modifier),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(2.dp)
    ) {
        if (thumbnail != null) {
            // Show image only (tap to open full-screen viewer)
            Image(
                bitmap = thumbnail.asImageBitmap(),
                contentDescription = "Search result image",
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 120.dp, max = 240.dp)
                    .clip(RoundedCornerShape(12.dp)),
                contentScale = ContentScale.Fit
            )
        } else {
            // No in-session thumbnail — show minimal file info from DB record
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 16.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .background(
                            meta?.circleColor ?: AppBackground,
                            RoundedCornerShape(10.dp)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        meta?.icon ?: Icons.Default.Description,
                        contentDescription = null,
                        tint = meta?.headerColor ?: AppAccent,
                        modifier = Modifier.size(24.dp)
                    )
                }
                Column(Modifier.weight(1f)) {
                    Text(
                        result.sourcePath.substringAfterLast("/"),
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = AppTitleColor,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Spacer(Modifier.height(2.dp))
                    Text(
                        meta?.displayName ?: result.subject,
                        fontSize = 11.sp,
                        color = meta?.headerColor ?: AppAccent,
                        fontWeight = FontWeight.Medium
                    )
                }
                Icon(
                    Icons.Default.ImageNotSupported,
                    contentDescription = "No preview",
                    tint = Color(0xFFCCCCCC),
                    modifier = Modifier.size(18.dp)
                )
            }
        }
    }
}

// ─── Full-screen Image Viewer Dialog (zoomable + pannable + closeable) ────────

@Composable
fun ImageViewerDialog(image: Bitmap, onDismiss: () -> Unit) {
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }

    val transformState = rememberTransformableState { zoomChange, panChange, _ ->
        scale = (scale * zoomChange).coerceIn(0.5f, 6f)
        offset += panChange * scale
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,
            dismissOnBackPress = true,
            dismissOnClickOutside = false
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
        ) {
            // Zoomable & pannable image
            Image(
                bitmap = image.asImageBitmap(),
                contentDescription = "Full size image",
                modifier = Modifier
                    .fillMaxSize()
                    .graphicsLayer(
                        scaleX = scale,
                        scaleY = scale,
                        translationX = offset.x,
                        translationY = offset.y
                    )
                    .transformable(state = transformState),
                contentScale = ContentScale.Fit
            )

            // Close button (top-right)
            IconButton(
                onClick = onDismiss,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
                    .size(40.dp)
                    .background(Color.Black.copy(alpha = 0.55f), CircleShape)
            ) {
                Icon(
                    Icons.Default.Close,
                    contentDescription = "Close",
                    tint = Color.White,
                    modifier = Modifier.size(22.dp)
                )
            }

            // Hint text (fades away after first zoom)
            if (scale == 1f) {
                Text(
                    "Pinch to zoom • Drag to pan",
                    fontSize = 11.sp,
                    color = Color.White.copy(alpha = 0.6f),
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 20.dp)
                        .background(Color.Black.copy(alpha = 0.4f), RoundedCornerShape(20.dp))
                        .padding(horizontal = 12.dp, vertical = 4.dp)
                )
            }
        }
    }
}