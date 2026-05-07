# Gemma 4 E2B-it ONNX Export Notes

The Android module `:gemmaimage` is ready to import a Gemma ONNX bundle, but
the ONNX graph files are not produced in this repo yet.

This direct image pipeline is not a normal single-output classifier:

```text
image -> Gemma multimodal encoder -> autoregressive text decoder -> JSON parser
```

For Android, that usually needs multiple generation graphs plus token stepping:

- `vision_encoder.onnx`
- `text_embedder.onnx`
- `decoder_model.onnx`
- `decoder_with_past_model.onnx`
- tokenizer / processor / chat-template sidecars

The current Python pipeline in `OnDeviceAI/Complete_model` uses Hugging Face
`AutoProcessor` + `AutoModelForMultimodalLM.generate(...)`. Exporting
`generate(...)` directly as one ONNX file is not the right shape for mobile.

After ONNX export exists, package it with:

```bash
cd /home/interns.2026/slmmodels/GoogleApp

python3 tools/prepare_gemma_android_bundle.py \
  --hf-model-dir ../OnDeviceAI/models/base_models/gemma-4-e2b-it \
  --onnx-dir /path/to/exported/gemma4_onnx \
  --bundle-dir gemmaimage/src/main/assets/gemma_image/gemma4_e2b_it_onnx
```

Then use the Android API:

```kotlin
val pipeline = GemmaImagePipeline(context)
val result = pipeline.classify(bitmap)
```

Until ONNX files are present, use `bundleStatus()` to show missing files and
`parseGeneratedResponse(...)` to test UI parsing with raw JSON returned by the
Python Gemma pipeline.
