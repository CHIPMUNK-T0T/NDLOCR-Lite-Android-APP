"""Run OCR on 3 sample images for FP32/QInt4/QInt8/QInt16 PARSEQ weights.

Outputs per image and per variant:
- processing times
- OCR text
- line-level bounding boxes
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from onnxruntime.quantization import QuantType, quantize_dynamic
from yaml import safe_load

import sys


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "ndlocr-lite" / "src"
MODEL_DIR = SRC_DIR / "model"
CONFIG_DIR = SRC_DIR / "config"

SAMPLE_IMAGES = [
    ROOT / "ndlocr-lite" / "resource" / "digidepo_2531162_0024.jpg",
    ROOT / "ndlocr-lite" / "resource" / "digidepo_3048008_0025.jpg",
    ROOT / "ndlocr-lite" / "resource" / "digidepo_11048278_po_geppo1803_00021.jpg",
]

OUT_DIR = ROOT / "model_eval" / "weight_compare_results"
QUANT_MODEL_DIR = OUT_DIR / "quant_models"

sys.path.append(str(SRC_DIR))
from deim import DEIM  # noqa: E402
from parseq import PARSEQ  # noqa: E402


@dataclass
class LineRegion:
    idx: int
    x: int
    y: int
    w: int
    h: int
    pred_char_cnt: float
    crop: np.ndarray


@dataclass
class LineResult:
    idx: int
    bbox: list[int]
    pred_char_cnt: float
    text: str


@dataclass
class OcrResult:
    image: str
    variant: str
    ok: bool
    error: Optional[str]
    times_sec: dict[str, float]
    line_count: int
    lines: list[LineResult]
    full_text: str


def safe_quantize(src: Path, dst: Path, weight_type: QuantType) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    safe_tmp = ROOT / "model_eval" / ".tmp_ort_quant"
    safe_tmp.mkdir(parents=True, exist_ok=True)
    prev_tmp = os.environ.get("TMP")
    prev_temp = os.environ.get("TEMP")
    tempfile.tempdir = str(safe_tmp)
    os.environ["TMP"] = str(safe_tmp)
    os.environ["TEMP"] = str(safe_tmp)
    try:
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            op_types_to_quantize=["MatMul", "Gemm"],
            weight_type=weight_type,
            per_channel=True,
        )
    finally:
        tempfile.tempdir = None
        if prev_tmp is None:
            os.environ.pop("TMP", None)
        else:
            os.environ["TMP"] = prev_tmp
        if prev_temp is None:
            os.environ.pop("TEMP", None)
        else:
            os.environ["TEMP"] = prev_temp


def load_charlist() -> list[str]:
    cfg = safe_load((CONFIG_DIR / "NDLmoji.yaml").read_text(encoding="utf-8"))
    return list(cfg["model"]["charset_train"])


def make_detector() -> DEIM:
    return DEIM(
        model_path=str(MODEL_DIR / "deim-s-1024x1024.onnx"),
        class_mapping_path=str(CONFIG_DIR / "ndl.yaml"),
        score_threshold=0.2,
        conf_threshold=0.25,
        iou_threshold=0.2,
        device="cpu",
    )


def detect_line_regions(detector: DEIM, image_path: Path) -> tuple[list[LineRegion], dict[str, float]]:
    img = np.array(Image.open(image_path).convert("RGB"))
    t0 = time.perf_counter()
    detections = detector.detect(img)
    det_sec = time.perf_counter() - t0

    lines: list[LineRegion] = []
    idx = 0
    for det in detections:
        if det["class_index"] != 0:
            continue
        xmin, ymin, xmax, ymax = det["box"]
        xmin = max(0, int(xmin))
        ymin = max(0, int(ymin))
        xmax = min(img.shape[1], int(xmax))
        ymax = min(img.shape[0], int(ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        crop = img[ymin:ymax, xmin:xmax, :]
        pcc = float(det.get("pred_char_count", 100.0))
        lines.append(
            LineRegion(
                idx=idx,
                x=xmin,
                y=ymin,
                w=xmax - xmin,
                h=ymax - ymin,
                pred_char_cnt=pcc,
                crop=crop,
            )
        )
        idx += 1

    # Typical vertical Japanese pages: right-to-left columns.
    lines.sort(key=lambda ln: (-ln.x, ln.y))
    for i, ln in enumerate(lines):
        ln.idx = i
    return lines, {"detector_infer": det_sec}


def load_recognizers(rec30: Path, rec50: Path, rec100: Path, charlist: list[str]) -> tuple[PARSEQ, PARSEQ, PARSEQ, float]:
    t0 = time.perf_counter()
    r30 = PARSEQ(model_path=str(rec30), charlist=charlist, device="cpu")
    r50 = PARSEQ(model_path=str(rec50), charlist=charlist, device="cpu")
    r100 = PARSEQ(model_path=str(rec100), charlist=charlist, device="cpu")
    return r30, r50, r100, time.perf_counter() - t0


def run_cascade(lines: list[LineRegion], r30: PARSEQ, r50: PARSEQ, r100: PARSEQ) -> tuple[list[LineResult], float]:
    t0 = time.perf_counter()

    target30: list[LineRegion] = []
    target50: list[LineRegion] = []
    target100: list[LineRegion] = []
    for ln in lines:
        if ln.pred_char_cnt == 3:
            target30.append(ln)
        elif ln.pred_char_cnt == 2:
            target50.append(ln)
        else:
            target100.append(ln)

    output: dict[int, LineResult] = {}
    for ln in target30:
        pred = r30.read(ln.crop)
        if len(pred) >= 25:
            target50.append(ln)
        else:
            output[ln.idx] = LineResult(ln.idx, [ln.x, ln.y, ln.x + ln.w, ln.y + ln.h], ln.pred_char_cnt, pred)
    for ln in target50:
        pred = r50.read(ln.crop)
        if len(pred) >= 45:
            target100.append(ln)
        else:
            output[ln.idx] = LineResult(ln.idx, [ln.x, ln.y, ln.x + ln.w, ln.y + ln.h], ln.pred_char_cnt, pred)
    for ln in target100:
        pred = r100.read(ln.crop)
        output[ln.idx] = LineResult(ln.idx, [ln.x, ln.y, ln.x + ln.w, ln.y + ln.h], ln.pred_char_cnt, pred)

    ordered = [output[k] for k in sorted(output.keys())]
    return ordered, time.perf_counter() - t0


def save_result(out_base: Path, result: OcrResult) -> None:
    out_base.mkdir(parents=True, exist_ok=True)
    img_stem = Path(result.image).stem
    out_json = out_base / f"{img_stem}.json"
    out_txt = out_base / f"{img_stem}.txt"
    out_json.write_text(
        json.dumps(
            {
                **asdict(result),
                "lines": [asdict(ln) for ln in result.lines],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    out_txt.write_text(result.full_text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QUANT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    charlist = load_charlist()
    detector = make_detector()

    fp32_paths = (
        MODEL_DIR / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx",
        MODEL_DIR / "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx",
        MODEL_DIR / "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx",
    )

    variant_paths: dict[str, tuple[Path, Path, Path]] = {"fp32": fp32_paths}
    variant_errors: dict[str, str] = {}

    quant_variants = [
        ("qint4", QuantType.QInt4),
        ("qint8", QuantType.QInt8),
        ("qint16", QuantType.QInt16),
    ]
    for name, qtype in quant_variants:
        try:
            q30 = QUANT_MODEL_DIR / f"{fp32_paths[0].stem}_{name}.onnx"
            q50 = QUANT_MODEL_DIR / f"{fp32_paths[1].stem}_{name}.onnx"
            q100 = QUANT_MODEL_DIR / f"{fp32_paths[2].stem}_{name}.onnx"
            safe_quantize(fp32_paths[0], q30, qtype)
            safe_quantize(fp32_paths[1], q50, qtype)
            safe_quantize(fp32_paths[2], q100, qtype)
            variant_paths[name] = (q30, q50, q100)
        except Exception as e:
            variant_errors[name] = f"quantize_failed: {e}"

    summary: dict[str, dict] = {}
    for image_path in SAMPLE_IMAGES:
        if not image_path.exists():
            continue
        line_regions, det_times = detect_line_regions(detector, image_path)
        summary[image_path.name] = {}

        for variant in ["fp32", "qint4", "qint8", "qint16"]:
            variant_out = OUT_DIR / variant
            if variant in variant_errors:
                res = OcrResult(
                    image=str(image_path),
                    variant=variant,
                    ok=False,
                    error=variant_errors[variant],
                    times_sec=det_times,
                    line_count=0,
                    lines=[],
                    full_text="",
                )
                save_result(variant_out, res)
                summary[image_path.name][variant] = {"ok": False, "error": variant_errors[variant]}
                continue

            rec30, rec50, rec100 = variant_paths[variant]
            try:
                r30, r50, r100, load_sec = load_recognizers(rec30, rec50, rec100, charlist)
                lines, infer_sec = run_cascade(line_regions, r30, r50, r100)
                full_text = "\n".join([ln.text for ln in lines])
                times = {
                    **det_times,
                    "recognizer_load": load_sec,
                    "recognizer_infer": infer_sec,
                    "total": det_times["detector_infer"] + load_sec + infer_sec,
                }
                res = OcrResult(
                    image=str(image_path),
                    variant=variant,
                    ok=True,
                    error=None,
                    times_sec=times,
                    line_count=len(lines),
                    lines=lines,
                    full_text=full_text,
                )
                save_result(variant_out, res)
                summary[image_path.name][variant] = {
                    "ok": True,
                    "line_count": len(lines),
                    "times_sec": times,
                }
            except Exception as e:
                res = OcrResult(
                    image=str(image_path),
                    variant=variant,
                    ok=False,
                    error=str(e),
                    times_sec=det_times,
                    line_count=0,
                    lines=[],
                    full_text="",
                )
                save_result(variant_out, res)
                summary[image_path.name][variant] = {"ok": False, "error": str(e)}

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved results under: {OUT_DIR}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
