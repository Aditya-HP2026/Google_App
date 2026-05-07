# Gemma Image Android Library

Module: `:gemmaimage`

Purpose: Android-side wrapper for a future Gemma 4 E2B-it ONNX bundle that
takes an image and returns:

- label: `maths`, `physics`, `chemistry`, `biology`
- confidence
- keywords
- evidence text

The public API is:

```kotlin
val pipeline = GemmaImagePipeline(context)
val status = pipeline.bundleStatus()

// Once ONNX generation graphs are bundled:
// val result = pipeline.classify(bitmap)

// For UI tests with saved Python output:
val result = pipeline.parseGeneratedResponse(rawGemmaJson)
```

## Asset Bundle

Assets live at:

```text
gemmaimage/src/main/assets/gemma_image/gemma4_e2b_it_onnx/
```

Required sidecars:

- `tokenizer.json`
- `tokenizer_config.json`
- `processor_config.json`
- `generation_config.json`
- `chat_template.jinja`
- `prompt.txt`

Expected ONNX files:

- `vision_encoder.onnx`
- `text_embedder.onnx`
- `decoder_model.onnx`
- `decoder_with_past_model.onnx`

## Prepare Bundle

```bash
cd /home/interns.2026/slmmodels/GoogleApp

python3 tools/prepare_gemma_android_bundle.py \
  --hf-model-dir ../OnDeviceAI/models/base_models/gemma-4-e2b-it \
  --onnx-dir /path/to/exported/gemma4_onnx \
  --bundle-dir gemmaimage/src/main/assets/gemma_image/gemma4_e2b_it_onnx
```

If you do not have ONNX files yet, omit `--onnx-dir`; the tool will still copy
the tokenizer/processor sidecars and keep the bundle marked as not ready.
