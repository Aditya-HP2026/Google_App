#!/usr/bin/env python3
"""Prepare the Android asset folder for Gemma 4 E2B-it ONNX deployment.

This script does not magically convert Gemma to ONNX. It packages files that
Android needs after an ONNX export exists:

1. Hugging Face processor/tokenizer sidecars.
2. Exported ONNX graph files.
3. A manifest consumed by the :gemmaimage Android library.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SIDECAR_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "generation_config.json",
    "chat_template.jinja",
]

EXPECTED_ONNX_FILES = [
    "vision_encoder.onnx",
    "text_embedder.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
]

LABELS = ["maths", "physics", "chemistry", "biology"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package Gemma 4 E2B-it sidecars and exported ONNX graphs for Android Studio.",
    )
    parser.add_argument(
        "--hf-model-dir",
        type=Path,
        default=Path("../OnDeviceAI/models/base_models/gemma-4-e2b-it"),
        help="Local Hugging Face Gemma 4 E2B-it directory.",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=None,
        help="Directory containing exported ONNX graph files.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("gemmaimage/src/main/assets/gemma_image/gemma4_e2b_it_onnx"),
        help="Android asset bundle directory to populate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_model_dir = args.hf_model_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    onnx_dir = args.onnx_dir.resolve() if args.onnx_dir else None

    if not hf_model_dir.exists():
        raise FileNotFoundError(f"HF model directory not found: {hf_model_dir}")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied_sidecars = copy_files(hf_model_dir, bundle_dir, SIDECAR_FILES, required=True)
    copied_onnx: list[str] = []
    missing_onnx = list(EXPECTED_ONNX_FILES)

    if onnx_dir is not None:
        if not onnx_dir.exists():
            raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")
        copied_onnx = copy_files(onnx_dir, bundle_dir, EXPECTED_ONNX_FILES, required=False)
        missing_onnx = [name for name in EXPECTED_ONNX_FILES if name not in copied_onnx]

    prompt_path = bundle_dir / "prompt.txt"
    if not prompt_path.exists():
        prompt_path.write_text(default_prompt(), encoding="utf-8")

    manifest = {
        "name": "gemma4_e2b_it_image_subject_keywords",
        "model": "google/gemma-4-E2B-it",
        "status": "ready" if not missing_onnx else "onnx_export_required",
        "bundle_version": 1,
        "runtime": "onnxruntime-android",
        "task": "direct_image_to_subject_keywords",
        "labels": LABELS,
        "prompt_file": "prompt.txt",
        "required_sidecars": SIDECAR_FILES,
        "expected_onnx_files": EXPECTED_ONNX_FILES,
        "copied_sidecars": copied_sidecars,
        "copied_onnx_files": copied_onnx,
        "missing_onnx_files": missing_onnx,
    }
    (bundle_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Bundle directory: {bundle_dir}")
    print(f"Copied sidecars: {', '.join(copied_sidecars)}")
    if copied_onnx:
        print(f"Copied ONNX files: {', '.join(copied_onnx)}")
    if missing_onnx:
        print(f"Missing ONNX files: {', '.join(missing_onnx)}")
        print("Bundle is prepared but not runnable until ONNX graph files are added.")
    else:
        print("Bundle is ready.")


def copy_files(source_dir: Path, target_dir: Path, filenames: list[str], required: bool) -> list[str]:
    copied: list[str] = []
    for name in filenames:
        source = source_dir / name
        if not source.exists():
            if required:
                raise FileNotFoundError(f"Required file missing: {source}")
            continue
        shutil.copy2(source, target_dir / name)
        copied.append(name)
    return copied


def default_prompt() -> str:
    return (
        "Analyze this academic note image directly. Read handwriting, formulas, diagrams, "
        "and visible text yourself. Choose exactly one subject from: maths, physics, "
        "chemistry, biology. Extract 5 to 12 useful keywords from the image. Return only "
        "valid JSON with this schema: "
        '{"subject":"chemistry","confidence":0.0,"keywords":["keyword"],"evidence":"short reason"}. '
        "The confidence must be a number from 0 to 1.\n"
    )


if __name__ == "__main__":
    main()
