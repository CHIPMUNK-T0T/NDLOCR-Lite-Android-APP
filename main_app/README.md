# main_app

Minimal Android app skeleton for NDLOCR-Lite style pipeline.

## Current status
- CameraX preview + ImageAnalysis
- latest-only frame store (`KEEP_ONLY_LATEST` + `LatestFrameStore`)
- dual-rate orchestrator
  - detector interval default: 5000ms
  - recognizer interval default: 2500ms
- overlay drawing for ROIs/text
- metrics view (detector ms / recognizer ms / dropped frames)
- ONNX Runtime FP32 engines wired
  - detector: `deim-s-1024x1024.onnx`
  - recognizer: `parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx`

## Package layout
- `com.ndl.mainapp.MainActivity`
- `com.ndl.mainapp.ocr.CameraPipeline`
- `com.ndl.mainapp.ocr.LatestFrameStore`
- `com.ndl.mainapp.ocr.OcrOrchestrator`
- `com.ndl.mainapp.ocr.DetectorEngine` (interface)
- `com.ndl.mainapp.ocr.RecognizerEngine` (interface)
- `com.ndl.mainapp.ocr.ui.OverlayView`

## Model runtime note
- MainActivity first tries ORT engines.
- If model load fails, it falls back to `NoopDetectorEngine` / `NoopRecognizerEngine`.

## Recommended asset paths
Put model files here:
- `app/src/main/assets/models/deim-s-1024x1024.onnx`
- `app/src/main/assets/models/parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx`

## Build
Open `main_app/` with Android Studio and sync Gradle.
