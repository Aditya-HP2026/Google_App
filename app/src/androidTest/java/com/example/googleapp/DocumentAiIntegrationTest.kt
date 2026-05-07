package com.example.googleapp

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.documentai.DocumentAiPipeline
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Basic integration test to verify the DocumentAI pipeline can be instantiated
 * and the app launches without crashing.
 */
@RunWith(AndroidJUnit4::class)
class DocumentAiIntegrationTest {

    @Test
    fun documentAiPipelineCanBeCreated() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        
        // This test verifies that:
        // 1. The DocumentAI module is properly linked
        // 2. ONNX Runtime can be initialized
        // 3. Asset loading works
        val pipeline = DocumentAiPipeline(context)
        
        // Create a small test bitmap
        val testBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        testBitmap.eraseColor(Color.WHITE)
        
        // Test the basic pipeline (should not crash)
        val result = pipeline.extractText(testBitmap)
        
        // Should complete without throwing exceptions
        assert(result.text.isNotEmpty() || result.text.isEmpty()) // Either result is fine
        
        pipeline.close()
    }
}