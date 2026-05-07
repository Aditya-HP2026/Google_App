# Gemma 4 E2B-it ONNX Asset Bundle

Put the exported ONNX generation bundle files in this folder before building the APK.

Expected runtime files:

- `vision_encoder.onnx`
- `text_embedder.onnx`
- `decoder_model.onnx`
- `decoder_with_past_model.onnx`

Expected sidecar files copied from the Hugging Face model:

- `tokenizer.json`
- `tokenizer_config.json`
- `processor_config.json`
- `generation_config.json`
- `chat_template.jinja`

Use:

```bash
cd /home/interns.2026/slmmodels/GoogleApp
python3 tools/prepare_gemma_android_bundle.py \
  --hf-model-dir ../OnDeviceAI/models/base_models/gemma-4-e2b-it \
  --onnx-dir /path/to/exported/gemma4_onnx \
  --bundle-dir gemmaimage/src/main/assets/gemma_image/gemma4_e2b_it_onnx
```
