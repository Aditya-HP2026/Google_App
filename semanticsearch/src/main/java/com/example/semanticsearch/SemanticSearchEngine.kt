package com.example.semanticsearch

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import com.example.semanticsearch.model.SearchResult
import org.json.JSONArray
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.min

/**
 * Semantic search engine backed by a SQLite knowledge.db.
 *
 * Usage:
 * ```kotlin
 * val engine = SemanticSearchEngine(context, File(getExternalFilesDir(null), "knowledge.db"))
 * 
 * // Index a new document
 * engine.indexDocument("photo_001.jpg", "chemistry", listOf("ionic", "covalent"), "The image shows...")
 * 
 * // Search
 * val results = engine.search("chemical bonding ionic covalent", topK = 5)
 * engine.close()
 * ```
 *
 * The knowledge.db schema:
 *   documents(id, source_path, page_num, subject, keywords TEXT,
 *             text_chunk, embedding BLOB, embedding_dim, created_at)
 *
 * Embeddings are stored as little-endian float32 BLOBs (embedding_dim = 384).
 */
class SemanticSearchEngine(
    context: Context,
    private val dbFile: File
) : AutoCloseable {

    private val encoder = MiniLMEncoder(context)
    private val db: SQLiteDatabase
    private val lock = Any() // Thread-safety lock for encoder + db access

    companion object {
        private const val EMBEDDING_DIM = 384

        private const val CREATE_TABLE_SQL = """
            CREATE TABLE IF NOT EXISTS documents (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path   TEXT    NOT NULL,
                page_num      INTEGER NOT NULL DEFAULT 0,
                subject       TEXT,
                keywords      TEXT,
                text_chunk    TEXT,
                embedding     BLOB    NOT NULL,
                embedding_dim INTEGER NOT NULL DEFAULT 384,
                created_at    TEXT    NOT NULL
            )
        """

        private const val CREATE_INDEX_SQL = """
            CREATE INDEX IF NOT EXISTS idx_subject ON documents(subject)
        """
    }

    init {
        // Create or open database in read/write mode
        db = if (dbFile.exists()) {
            SQLiteDatabase.openDatabase(dbFile.absolutePath, null, SQLiteDatabase.OPEN_READWRITE)
        } else {
            // Create new database
            dbFile.parentFile?.mkdirs()
            SQLiteDatabase.openOrCreateDatabase(dbFile, null).also { database ->
                database.execSQL(CREATE_TABLE_SQL)
                database.execSQL(CREATE_INDEX_SQL)
            }
        }
    }

    /**
     * Index a new document for semantic search.
     *
     * @param sourcePath File path or identifier for the document
     * @param subject The classified subject (e.g., "chemistry")
     * @param keywords List of extracted keywords
     * @param textChunk Text content or evidence from classification
     * @param pageNum Optional page number (default 0)
     * @return The row ID of the inserted document
     */
    fun indexDocument(
        sourcePath: String,
        subject: String,
        keywords: List<String>,
        textChunk: String,
        pageNum: Int = 0
    ): Long = synchronized(lock) {
        // Build text for embedding: combine subject, keywords, and text
        val embeddingText = buildString {
            append(subject)
            append(" | ")
            append(keywords.joinToString(" "))
            append(" | ")
            append(textChunk.take(500))
        }

        // Generate embedding
        val embedding = encoder.encode(embeddingText)
        val embeddingBlob = floatArrayToBlob(embedding)

        // Insert into database
        val values = ContentValues().apply {
            put("source_path", sourcePath)
            put("page_num", pageNum)
            put("subject", subject)
            put("keywords", JSONArray(keywords).toString())
            put("text_chunk", textChunk)
            put("embedding", embeddingBlob)
            put("embedding_dim", EMBEDDING_DIM)
            put("created_at", SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(Date()))
        }

        db.insert("documents", null, values)
    }

    /**
     * Returns the top-[topK] documents most semantically similar to [query].
     * Optionally filter by [subjectFilter] (e.g. "chemistry").
     */
    fun search(
        query: String,
        topK: Int = 5,
        subjectFilter: String? = null
    ): List<SearchResult> = synchronized(lock) {
        val queryEmbedding = encoder.encode(query)

        val sql = if (subjectFilter != null) {
            "SELECT source_path, page_num, subject, keywords, text_chunk, embedding " +
            "FROM documents WHERE subject = ?"
        } else {
            "SELECT source_path, page_num, subject, keywords, text_chunk, embedding " +
            "FROM documents"
        }

        val args = if (subjectFilter != null) arrayOf(subjectFilter) else null
        val cursor = db.rawQuery(sql, args)

        data class Candidate(
            val sourcePath: String,
            val pageNum: Int,
            val subject: String,
            val keywords: List<String>,
            val textChunk: String,
            val score: Float
        )

        val candidates = mutableListOf<Candidate>()

        cursor.use { c ->
            val colSource   = c.getColumnIndexOrThrow("source_path")
            val colPage     = c.getColumnIndexOrThrow("page_num")
            val colSubject  = c.getColumnIndexOrThrow("subject")
            val colKeywords = c.getColumnIndexOrThrow("keywords")
            val colText     = c.getColumnIndexOrThrow("text_chunk")
            val colEmb      = c.getColumnIndexOrThrow("embedding")

            while (c.moveToNext()) {
                val blob = c.getBlob(colEmb) ?: continue
                val docEmbedding = blobToFloatArray(blob)
                if (docEmbedding.size != queryEmbedding.size) continue

                val score = cosineSimilarity(queryEmbedding, docEmbedding)
                val keywordsJson = c.getString(colKeywords) ?: "[]"
                val keywords = parseKeywords(keywordsJson)

                candidates += Candidate(
                    sourcePath = c.getString(colSource) ?: "",
                    pageNum    = c.getInt(colPage),
                    subject    = c.getString(colSubject) ?: "",
                    keywords   = keywords,
                    textChunk  = c.getString(colText) ?: "",
                    score      = score
                )
            }
        }

        candidates.sortByDescending { it.score }

        candidates.subList(0, min(topK, candidates.size)).map {
            SearchResult(
                sourcePath = it.sourcePath,
                pageNum    = it.pageNum,
                subject    = it.subject,
                keywords   = it.keywords,
                textChunk  = it.textChunk,
                score      = it.score
            )
        }
    }

    /** Returns true if knowledge.db has at least one indexed document. */
    fun hasDocuments(): Boolean {
        val c = db.rawQuery("SELECT COUNT(*) FROM documents", null)
        return c.use { it.moveToFirst() && it.getInt(0) > 0 }
    }

    /** Returns the total number of indexed documents. */
    fun documentCount(): Int {
        val c = db.rawQuery("SELECT COUNT(*) FROM documents", null)
        return c.use { if (it.moveToFirst()) it.getInt(0) else 0 }
    }

    private fun floatArrayToBlob(arr: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(arr.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        buffer.asFloatBuffer().put(arr)
        return buffer.array()
    }

    private fun blobToFloatArray(blob: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val arr = FloatArray(buf.remaining())
        buf.get(arr)
        return arr
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        // Both vectors are already L2-normalized at index time
        // so cosine similarity = dot product
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot
    }

    private fun parseKeywords(json: String): List<String> = try {
        val arr = JSONArray(json)
        List(arr.length()) { arr.getString(it) }
    } catch (_: Exception) { emptyList() }

    override fun close() {
        encoder.close()
        db.close()
    }
}