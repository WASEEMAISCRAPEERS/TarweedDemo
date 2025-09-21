"""
FastAPI app to serve Ultralytics YOLO model `terweejV1.pt` (default) for image inference
and a robust barcode-scanner endpoint `backImage` using pyzbar + OpenCV.

Run:
  uvicorn app:app --host 0.0.0.0 --port 8018 --reload

Env vars (optional):
  MODEL_PATH=/absolute/or/relative/path/to/model.pt
  DEVICE=cpu  (or "mps" on Apple Silicon, or a CUDA index like "0")
  APP_PORT=8018
  PRODUCTS_JSON=products.json

Deps:
  pip install ultralytics fastapi uvicorn pillow numpy opencv-python pyzbar python-dotenv
  # macOS may need system lib: brew install zbar
"""

from __future__ import annotations
import io
import os
import time
import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Ultralytics
from ultralytics import YOLO

# Barcode libs
from pyzbar.pyzbar import decode as zbar_decode
import cv2

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

APP_NAME = "tarweej-yolo-api"
APP_PORT = int(os.getenv("APP_PORT", 8018))
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "terweejV1.pt")
DEFAULT_DEVICE = os.getenv("DEVICE", None)  # e.g., "cpu", "mps", "0"
PRODUCTS_JSON_PATH = os.getenv("PRODUCTS_JSON", "products.json")

app = FastAPI(title=APP_NAME, version="1.5.0")

# --- CORS: allow all origins/methods/headers and expose X-Barcodes ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow any origin
    allow_credentials=False,      # keep False when using wildcard origins
    allow_methods=["*"],          # allow all HTTP methods
    allow_headers=["*"],          # allow all request headers
    expose_headers=["X-Barcodes"] # make this readable by browsers
)

# --- Load model once at startup ---
try:
    yolo_model = YOLO(DEFAULT_MODEL_PATH)
    if DEFAULT_DEVICE:
        _ = yolo_model.to(DEFAULT_DEVICE)
except Exception as e:
    yolo_model = None
    app.state.model_load_error = str(e)
else:
    app.state.model_load_error = None

# --- Load products DB once if available ---
if os.path.exists(PRODUCTS_JSON_PATH):
    try:
        with open(PRODUCTS_JSON_PATH, "r", encoding="utf-8") as f:
            PRODUCTS_DB = json.load(f)
    except Exception:
        PRODUCTS_DB = {}
else:
    PRODUCTS_DB = {}


class Detection(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    conf: float
    cls: int
    label: str

class PredictResponse(BaseModel):
    time_ms: float
    width: int
    height: int
    detections: List[Detection]


def _read_image_to_pil(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _result_to_json(result) -> Dict[str, Any]:
    boxes = result.boxes
    detections: List[Dict[str, Any]] = []
    names = getattr(result, "names", getattr(yolo_model, "names", {}))

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for i in range(xyxy.shape[0]):
            c = int(cls[i])
            label = str(names.get(c, str(c))) if isinstance(names, dict) else str(c)
            detections.append({
                "bbox": xyxy[i].tolist(),
                "conf": float(conf[i]),
                "cls": c,
                "label": label,
            })

    h, w = result.orig_shape if hasattr(result, "orig_shape") else (None, None)
    return {"width": w, "height": h, "detections": detections}


def _parse_bgr(csv: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    try:
        b, g, r = map(int, csv.split(","))
        return (b, g, r)
    except Exception:
        return fallback


def _draw_label(img: np.ndarray, text: str, x: int, y: int,
                bg_bgr: Tuple[int, int, int], txt_bgr: Tuple[int, int, int],
                scale: float = 0.6, thickness: int = 2, pad: int = 4) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top_left = (x, max(0, y - th - baseline - pad * 2))
    bottom_right = (x + tw + pad * 2, y)
    cv2.rectangle(img, top_left, bottom_right, bg_bgr, thickness=cv2.FILLED)
    cv2.putText(img, text, (x + pad, y - baseline - pad), cv2.FONT_HERSHEY_SIMPLEX, scale, txt_bgr, thickness, cv2.LINE_AA)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok" if (yolo_model is not None and app.state.model_load_error is None) else "error",
        "model_path": DEFAULT_MODEL_PATH,
        "model_loaded": yolo_model is not None,
        "model_error": app.state.model_load_error,
        "port": APP_PORT,
        "products_db_loaded": bool(PRODUCTS_DB),
        "products_json_path": PRODUCTS_JSON_PATH,
    }


# ---------------- YOLO Inference ---------------- #
@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold"),
    imgsz: int = Query(640, ge=64, le=4096, description="Inference image size"),
    device: Optional[str] = Query(None, description='Override device: "cpu", "mps", or CUDA index like "0"'),
) -> Any:
    if yolo_model is None:
        return JSONResponse(status_code=500, content={"error": app.state.model_load_error or "Model not loaded"})

    image = _read_image_to_pil(file)

    start = time.time()
    results = yolo_model(
        image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device or DEFAULT_DEVICE,
        verbose=False,
    )
    elapsed_ms = (time.time() - start) * 1000.0

    result = results[0]
    payload = _result_to_json(result)
    return PredictResponse(time_ms=elapsed_ms, **payload)


@app.post("/predict_image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    imgsz: int = Query(640, ge=64, le=4096),
    device: Optional[str] = Query(None),
    format: str = Query("jpg", pattern="^(jpg|png)$", description="Output image format"),
):
    if yolo_model is None:
        return JSONResponse(status_code=500, content={"error": app.state.model_load_error or "Model not loaded"})

    image = _read_image_to_pil(file)

    results = yolo_model(
        image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device or DEFAULT_DEVICE,
        verbose=False,
    )
    result = results[0]

    plotted = result.plot()  # BGR
    plotted_rgb = plotted[:, :, ::-1]
    pil_img = Image.fromarray(plotted_rgb)

    buf = io.BytesIO()
    mime = "image/jpeg" if format == "jpg" else "image/png"
    pil_img.save(buf, format="JPEG" if format == "jpg" else "PNG", quality=95)
    buf.seek(0)

    return StreamingResponse(buf, media_type=mime)


# ---------------- Barcode Utilities (Robust) ---------------- #

def _unique_by_data(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = (it.get("class"), it.get("data"))
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _try_pyzbar(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # Try color, gray, and binary
    variants = [img_bgr]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    variants.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
    for v in variants:
        for b in zbar_decode(v):
            data = b.data.decode("utf-8", errors="ignore")
            btype = b.type
            x, y, w, h = b.rect
            out.append({
                "class": btype,
                "data": data,
                "rect": {"x": x, "y": y, "w": w, "h": h},
                "source": "pyzbar",
            })
    return out


def _try_opencv_qr(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(img_bgr)
        if data and points is not None and len(points) == 4:
            xs = points[:, 0].astype(int)
            ys = points[:, 1].astype(int)
            x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max()-xs.min()), int(ys.max()-ys.min())
            out.append({
                "class": "QRCODE",
                "data": data,
                "rect": {"x": x, "y": y, "w": w, "h": h},
                "source": "opencv_qr",
            })
    except Exception:
        pass
    return out


def _roi_candidates(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Heuristic ROI finder for 1D codes using gradients + morphology."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.convertScaleAbs(cv2.subtract(gradX, gradY))
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois: List[Tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 500:
            continue
        ar = w / float(h)
        if ar < 1.5:
            continue
        rois.append((x, y, w, h))
    return rois


def _preprocess_variants(img_bgr: np.ndarray, upscale: int = 2) -> List[np.ndarray]:
    variants: List[np.ndarray] = []
    base = img_bgr.copy()
    h, w = base.shape[:2]
    variants.append(base)
    if max(h, w) < 1500:
        variants.append(cv2.resize(base, (w*upscale, h*upscale), interpolation=cv2.INTER_CUBIC))
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    variants.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
    den = cv2.bilateralFilter(base, 5, 50, 50)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    variants.append(cv2.filter2D(den, -1, kernel))
    return variants


def _rotations(img_bgr: np.ndarray) -> List[np.ndarray]:
    rots = [img_bgr]
    for k in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        rots.append(cv2.rotate(img_bgr, k))
    return rots


# ---------------- Barcode Endpoint (Robust) ---------------- #
@app.post("/backImage")
async def backImage(
    file: UploadFile = File(...),
    draw_color: str = Query("0,255,0", description="BGR rectangle color as 'B,G,R'"),
    text_color: str = Query("255,255,255", description="BGR text color as 'B,G,R'"),
    label_bg: str = Query("0,0,0", description="BGR filled background for class label"),
    wrap: int = Query(40, ge=10, le=120, description="Characters per wrapped text line"),
    format: str = Query("png", pattern="^(jpg|png)$", description="Output image format"),
):
    """Decode barcodes robustly (multi-rotate + multi-preprocess + ROI), draw bounding box + CLASS NAME, overlay data/product, return annotated image."""
    raw = await file.read()
    np_arr = np.frombuffer(raw, np.uint8)
    img0 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img0 is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image data"})

    box_bgr = _parse_bgr(draw_color, (0, 255, 0))
    txt_bgr = _parse_bgr(text_color, (255, 255, 255))
    lbl_bgr = _parse_bgr(label_bg, (0, 0, 0))

    found: List[Dict[str, Any]] = []
    # Global tries
    for rot in _rotations(img0):
        for var in _preprocess_variants(rot):
            found.extend(_try_pyzbar(var))
            found.extend(_try_opencv_qr(var))

    # ROI tries for 1D barcodes
    for (x, y, w, h) in _roi_candidates(img0):
        roi = img0[max(0, y - 10):y + h + 10, max(0, x - 10):x + w + 10]
        for var in _preprocess_variants(roi):
            hits = _try_pyzbar(var)
            # Map ROI rect back to full image coords
            for det in hits:
                r = det["rect"]
                det["rect"] = {"x": r["x"] + max(0, x - 10), "y": r["y"] + max(0, y - 10), "w": r["w"], "h": r["h"]}
            found.extend(hits)

    found = _unique_by_data(found)

    # Draw on original image
    img = img0.copy()
    for det in found:
        r = det["rect"]
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), box_bgr, 2)
        _draw_label(img, text=str(det.get("class", "BARCODE")), x=max(0, x), y=max(0, y),
                    bg_bgr=lbl_bgr, txt_bgr=txt_bgr, scale=0.6, thickness=2, pad=4)
        overlay = f"{det.get('data','')} | {PRODUCTS_DB.get(det.get('data',''), 'Not found in database')}"
        for i, line in enumerate(textwrap.wrap(overlay, width=wrap)):
            cv2.putText(img, line, (max(10, x), y + h + 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_bgr, 2, cv2.LINE_AA)

    # Encode output
    ext = ".jpg" if format == "jpg" else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return JSONResponse(status_code=500, content={"error": "Failed to encode output image"})

    bytes_io = io.BytesIO(buf.tobytes())
    mime = "image/jpeg" if format == "jpg" else "image/png"
    headers = {"X-Barcodes": json.dumps(found, ensure_ascii=False)}
    return StreamingResponse(bytes_io, media_type=mime, headers=headers)


@app.get("/")
async def root():
    return {"app": APP_NAME, "model": DEFAULT_MODEL_PATH, "port": APP_PORT}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=APP_PORT, reload=True)
