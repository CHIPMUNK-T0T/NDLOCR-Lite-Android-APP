"""Compare OCR on resource images for FP32 / QInt4 / QInt8 / QInt16.

Design:
- Run DEIM detector once (FP32) and reuse detected line crops.
- Quantize PARSEQ models by type (MatMul/Gemm only).
- Run cascade OCR for each variant.
- Report runtime and text drift vs FP32 baseline.
- Visualize detected edges and recognized text on output images.
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

# Use images from model_eval directory
_model_eval_imgs = list((ROOT / "model_eval").glob("digidepo*.jpg"))
INPUT_IMAGES = sorted(_model_eval_imgs) if _model_eval_imgs else []

WORK_DIR = ROOT / "model_eval" / "quant_ocr_runs"
QUANT_MODEL_DIR = WORK_DIR / "quant_models"
REPORT_JSON = WORK_DIR / "report_quant_types_ocr.json"
VIZ_OUTPUT_DIR = WORK_DIR / "visualizations"

sys.path.append(str(SRC_DIR))
from deim import DEIM  # noqa: E402
from parseq import PARSEQ  # noqa: E402


@dataclass
class RecogLine:
    npimg: np.ndarray
    idx: int
    pred_char_cnt: float
    x: int
    y: int
    pred_str: str = ""


@dataclass
class VariantResult:
    name: str
    ok: bool
    quant_error: Optional[str]
    model_load_sec: Optional[float]
    infer_sec: Optional[float]
    total_recog_sec: Optional[float]
    lines: int
    chars: int
    text: str
    line_exact_match_rate_vs_fp32: Optional[float]
    char_error_rate_vs_fp32: Optional[float]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


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


def detect_lines(input_image: Path) -> tuple[list[RecogLine], float]:
    t0 = time.perf_counter()
    detector = DEIM(
        model_path=str(MODEL_DIR / "deim-s-1024x1024.onnx"),
        class_mapping_path=str(CONFIG_DIR / "ndl.yaml"),
        score_threshold=0.2,
        conf_threshold=0.25,
        iou_threshold=0.2,
        device="cpu",
    )
    img = np.array(Image.open(input_image).convert("RGB"))
    detections = detector.detect(img)
    elapsed = time.perf_counter() - t0

    lines: list[RecogLine] = []
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
        lines.append(RecogLine(crop, idx, pcc, xmin, ymin))
        idx += 1

    # For Japanese vertical text pages, right-to-left columns are common.
    # Sort by x (desc) then y (asc) to stabilize output for comparison.
    lines.sort(key=lambda ln: (-ln.x, ln.y))
    for i, ln in enumerate(lines):
        ln.idx = i
    return lines, elapsed


def process_cascade(
    lines: list[RecogLine],
    recognizer30: PARSEQ,
    recognizer50: PARSEQ,
    recognizer100: PARSEQ,
) -> list[str]:
    target30: list[RecogLine] = []
    target50: list[RecogLine] = []
    target100: list[RecogLine] = []
    for ln in lines:
        if ln.pred_char_cnt == 3:
            target30.append(ln)
        elif ln.pred_char_cnt == 2:
            target50.append(ln)
        else:
            target100.append(ln)

    all_done: list[RecogLine] = []
    for ln in target30:
        pred = recognizer30.read(ln.npimg)
        if len(pred) >= 25:
            target50.append(ln)
        else:
            ln.pred_str = pred
            all_done.append(ln)
    for ln in target50:
        pred = recognizer50.read(ln.npimg)
        if len(pred) >= 45:
            target100.append(ln)
        else:
            ln.pred_str = pred
            all_done.append(ln)
    for ln in target100:
        ln.pred_str = recognizer100.read(ln.npimg)
        all_done.append(ln)

    all_done.sort(key=lambda x: x.idx)
    return [ln.pred_str for ln in all_done]


def run_variant(name: str, rec30: Path, rec50: Path, rec100: Path, lines: list[RecogLine], charlist: list[str]) -> VariantResult:
    try:
        t0 = time.perf_counter()
        r30 = PARSEQ(model_path=str(rec30), charlist=charlist, device="cpu")
        r50 = PARSEQ(model_path=str(rec50), charlist=charlist, device="cpu")
        r100 = PARSEQ(model_path=str(rec100), charlist=charlist, device="cpu")
        load_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        run_lines = [RecogLine(ln.npimg, ln.idx, ln.pred_char_cnt, ln.x, ln.y) for ln in lines]
        pred_lines = process_cascade(run_lines, r30, r50, r100)
        infer_sec = time.perf_counter() - t1

        text = "\n".join(pred_lines)
        return VariantResult(
            name=name,
            ok=True,
            quant_error=None,
            model_load_sec=load_sec,
            infer_sec=infer_sec,
            total_recog_sec=load_sec + infer_sec,
            lines=len(pred_lines),
            chars=len(text.replace("\n", "")),
            text=text,
            line_exact_match_rate_vs_fp32=None,
            char_error_rate_vs_fp32=None,
        )
    except Exception as e:
        return VariantResult(
            name=name,
            ok=False,
            quant_error=str(e),
            model_load_sec=None,
            infer_sec=None,
            total_recog_sec=None,
            lines=0,
            chars=0,
            text="",
            line_exact_match_rate_vs_fp32=None,
            char_error_rate_vs_fp32=None,
        )


def score_vs_fp32(base: VariantResult, target: VariantResult) -> None:
    if not base.ok or not target.ok:
        return
    b_lines = base.text.splitlines()
    t_lines = target.text.splitlines()
    n = max(len(b_lines), len(t_lines))
    exact = 0
    for i in range(n):
        bl = b_lines[i] if i < len(b_lines) else ""
        tl = t_lines[i] if i < len(t_lines) else ""
        if bl == tl:
            exact += 1
    target.line_exact_match_rate_vs_fp32 = exact / n if n > 0 else 1.0

    dist = levenshtein(base.text, target.text)
    denom = max(1, len(base.text))
    target.char_error_rate_vs_fp32 = dist / denom


def visualize_results(
    img: np.ndarray,
    lines: list[RecogLine],
    variant_result: VariantResult,
    output_path: Path,
) -> None:
    """Render detected edges and OCR text on the image."""
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil_img)

    # Color palette for visualization
    edge_color = (0, 255, 0)  # Green for edges
    text_bg_color = (0, 0, 255)  # Blue background for text
    text_color = (255, 255, 255)  # White text

    # Try Japanese fonts (Windows common fonts)
    font = None
    font_candidates = [
        "C:\\Windows\\Fonts\\msgothic.ttc",  # MS Gothic
        "C:\\Windows\\Fonts\\meiryo.ttc",    # Meiryo
        "C:\\Windows\\Fonts\\YuGothM.ttc",   # Yu Gothic
        "/usr/share/fonts/opentype/noto-cjk/NotoSansCJK-Regular.ttc",  # Linux
    ]
    
    for font_path in font_candidates:
        try:
            font = ImageFont.truetype(font_path, 12)
            break
        except Exception:
            continue
    
    if font is None:
        font = ImageFont.load_default()

    # Draw bounding boxes and text for each line
    for i, ln in enumerate(lines):
        if i < len(variant_result.text.splitlines()):
            text = variant_result.text.splitlines()[i]
        else:
            text = ""

        # Draw bounding box (edge)
        x1, y1 = ln.x, ln.y
        x2, y2 = ln.x + ln.npimg.shape[1], ln.y + ln.npimg.shape[0]
        draw.rectangle([x1, y1, x2, y2], outline=edge_color, width=2)

        # Draw recognized text above the box
        if text:
            text_bbox = draw.textbbox((x1, y1 - 20), text, font=font)
            text_width = text_bbox[2] - text_bbox[0] + 4
            text_height = text_bbox[3] - text_bbox[1] + 4
            # Draw background for text
            draw.rectangle(
                [x1 - 2, y1 - text_height - 2, x1 + text_width, y1 + 2],
                fill=text_bg_color,
            )
            # Draw text
            draw.text((x1, y1 - text_height), text, fill=text_color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(output_path))
    print(f"Visualization saved: {output_path}")


def main() -> None:
    if not INPUT_IMAGES:
        print("Error: No digidepo*.jpg images found in model_eval directory")
        return
    
    print(f"Found {len(INPUT_IMAGES)} images: {[img.name for img in INPUT_IMAGES]}")
    
    charlist = load_charlist()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    QUANT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_reports = {}

    for input_img in INPUT_IMAGES:
        img_name = input_img.stem
        print(f"\n{'='*60}")
        print(f"Processing: {img_name}")
        print(f"{'='*60}")
        
        lines, detector_sec = detect_lines(input_img)
        print(f"Detected {len(lines)} lines in {detector_sec:.4f}s")
        
        # Load original image for visualization
        orig_img = np.array(Image.open(input_img).convert("RGB"))
        
        # Create image-specific output directory
        img_viz_dir = VIZ_OUTPUT_DIR / img_name
        img_viz_dir.mkdir(parents=True, exist_ok=True)

        fp32 = run_variant(
            name="fp32",
            rec30=MODEL_DIR / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx",
            rec50=MODEL_DIR / "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx",
            rec100=MODEL_DIR / "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx",
            lines=lines,
            charlist=charlist,
        )
        results = [fp32]
        
        # Visualize FP32 results
        visualize_results(
            orig_img, 
            lines, 
            fp32, 
            img_viz_dir / "fp32_visualization.png"
        )
        print(f"[FP32] lines={fp32.lines} chars={fp32.chars} ok={fp32.ok}")

        variants = [
            ("qint8", QuantType.QInt8),
        ]

        recs = [
            MODEL_DIR / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx",
            MODEL_DIR / "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx",
            MODEL_DIR / "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx",
        ]

        for label, qtype in variants:
            qpaths = [QUANT_MODEL_DIR / f"{p.stem}_{label}.onnx" for p in recs]
            try:
                for src, dst in zip(recs, qpaths):
                    if not dst.exists():
                        safe_quantize(src, dst, qtype)
            except Exception as e:
                results.append(
                    VariantResult(
                        name=label,
                        ok=False,
                        quant_error=f"quantize_failed: {e}",
                        model_load_sec=None,
                        infer_sec=None,
                        total_recog_sec=None,
                        lines=0,
                        chars=0,
                        text="",
                        line_exact_match_rate_vs_fp32=None,
                        char_error_rate_vs_fp32=None,
                    )
                )
                continue

            res = run_variant(label, qpaths[0], qpaths[1], qpaths[2], lines, charlist)
            score_vs_fp32(fp32, res)
            results.append(res)
            
            # Visualize quantized results
            if res.ok:
                visualize_results(
                    orig_img, 
                    lines, 
                    res, 
                    img_viz_dir / f"{label}_visualization.png"
                )
            print(f"[{label:>6}] lines={res.lines} chars={res.chars} ok={res.ok} cer={res.char_error_rate_vs_fp32}")

        # Save report for this image
        report = {
            "input_image": str(input_img),
            "image_name": img_name,
            "detector_time_sec": detector_sec,
            "detected_lines": len(lines),
            "results": [asdict(r) for r in results],
        }
        all_reports[img_name] = report
        
        # Also save individual report for each image
        img_report_path = img_viz_dir / "report.json"
        img_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        print(f"Report saved: {img_report_path}")
        
        print(f"Visualizations saved to: {img_viz_dir}")

    # Save combined report
    combined_report = {
        "total_images": len(INPUT_IMAGES),
        "images": all_reports,
    }
    combined_report_path = VIZ_OUTPUT_DIR / "combined_report.json"
    combined_report_path.write_text(
        json.dumps(combined_report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\nCombined report saved: {combined_report_path}")


if __name__ == "__main__":
    main()
