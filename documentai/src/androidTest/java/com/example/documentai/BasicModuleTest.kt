package com.example.documentai

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BasicModuleTest {
    
    @Test
    fun contextIsAvailable() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assert(context != null)
        assert(context.packageName == "com.example.documentai")
    }
}