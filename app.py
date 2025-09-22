"""
TARWEEJ | Image & Barcode Detection Service (FastAPI)

- /predict         : YOLO detections as JSON
- /predict_image   : YOLO annotated image
- /backImage       : Barcode decode -> annotated image (green box, RED text)
- /backImage_json  : Barcode decode -> JSON only (no image)

Run:
  uvicorn app:app --host 0.0.0.0 --port 8018 --reload

Env vars (optional):
  MODEL_PATH=/absolute/or/relative/path/to/model.pt   (default: tarweejV3.pt)
  DEVICE=cpu | mps | "0"                              (CUDA index)
  APP_PORT=8018
  PRODUCTS_JSON=/Users/koraspond_developer/Desktop/tarweej/products.json

Deps:
  pip install ultralytics fastapi uvicorn pillow numpy opencv-python pyzbar python-dotenv
  # macOS needs system lib:
  #   brew install zbar
"""

from __future__ import annotations
import io
import os
import time
import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# YOLO (Ultralytics)
from ultralytics import YOLO

# Barcode decoder
from pyzbar.pyzbar import decode as zbar_decode

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

APP_NAME = "tarweej-yolo-barcode-api"
APP_PORT = int(os.getenv("APP_PORT", 8018))
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "tarweejV3.pt")
DEFAULT_DEVICE = os.getenv("DEVICE", None)  # e.g., "cpu", "mps", "0"
PRODUCTS_JSON_PATH = os.getenv(
    "PRODUCTS_JSON",
    "/Users/koraspond_developer/Desktop/tarweej/products.json"
)

app = FastAPI(title=APP_NAME, version="2.1.0")

# --- CORS: allow all & expose the X-Barcodes header ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Barcodes"],
)

# --- Load YOLO model once at startup ---
try:
    yolo_model = YOLO(DEFAULT_MODEL_PATH)
    if DEFAULT_DEVICE:
        _ = yolo_model.to(DEFAULT_DEVICE)
except Exception as e:
    yolo_model = None
    app.state.model_load_error = str(e)
else:
    app.state.model_load_error = None

# --- Load products DB once ---
if os.path.exists(PRODUCTS_JSON_PATH):
    try:
        with open(PRODUCTS_JSON_PATH, "r", encoding="utf-8") as f:
            PRODUCTS_DB: Dict[str, str] = json.load(f)
    except Exception:
        PRODUCTS_DB = {}
else:
    PRODUCTS_DB = {}


# ---------------- Models ---------------- #
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

class BarcodeBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class BarcodeDetection(BaseModel):
    cls: str
    data: str
    product: Optional[str] = None
    bbox: BarcodeBox

class BarcodeResponse(BaseModel):
    status: str
    image_width: int
    image_height: int
    count: int
    detections: List[BarcodeDetection]


# ---------------- Helpers ---------------- #
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


# ---------------- Health ---------------- #
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
    conf: float = Query(0.30, ge=0.0, le=1.0, description="Confidence threshold"),
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


# ---------------- Barcode Endpoints (pyzbar + OpenCV) ---------------- #
@app.post("/backImage")
async def backImage(
    file: UploadFile = File(...),
    draw_color: str = Query("0,255,0", description="BGR rectangle color 'B,G,R' (default green)"),
    text_color: str = Query("0,0,255", description="BGR text color 'B,G,R' (default red)"),
    wrap: int = Query(40, ge=10, le=120, description="Characters per wrapped text line"),
    format: str = Query("png", pattern="^(jpg|png)$", description="Output image format"),
):
    """
    Barcode -> annotated image. Also returns detections in `X-Barcodes` header.
    """
    raw = await file.read()
    np_arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image data"})

    box_bgr = _parse_bgr(draw_color, (0, 255, 0))   # green
    txt_bgr = _parse_bgr(text_color, (0, 0, 255))   # red

    barcodes = zbar_decode(img)
    found: List[Dict[str, Any]] = []

    if barcodes:
        for barcode in barcodes:
            barcode_data = (barcode.data.decode("utf-8")).strip()
            barcode_type = barcode.type

            # Lookup product in JSON database (exact key match)
            product_info = PRODUCTS_DB.get(barcode_data, "Not found in database")

            # Draw rectangle around barcode
            (x, y, w, h) = barcode.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), box_bgr, 2)

            # Combine barcode + product info
            overlay_text = f"{barcode_data} | {product_info}"

            # Image center position
            img_h, img_w = img.shape[:2]
            y_offset = img_h // 2  # vertical center

            for line in textwrap.wrap(overlay_text, width=wrap):
                # Get text size (smaller font)
                (text_w, text_h), baseline = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )

                # Center horizontally in the image
                text_x = (img_w // 2) - (text_w // 2)
                text_y = y_offset

                # Draw yellow background strip
                cv2.rectangle(
                    img,
                    (text_x - 8, text_y - text_h - 8),
                    (text_x + text_w + 8, text_y + baseline + 8),
                    (0, 255, 255),  # Yellow BGR
                    -1,
                )

                # Put text (black for contrast)
                cv2.putText(
                    img,
                    line,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,  # smaller font size
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                y_offset += text_h + 25  # line spacing

            found.append({
                "class": barcode_type,
                "data": barcode_data,
                "rect": {"x": x, "y": y, "w": w, "h": h},
                "product": product_info,
            })
    # Encode output image
    ext = ".jpg" if format == "jpg" else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return JSONResponse(status_code=500, content={"error": "Failed to encode output image"})

    bytes_io = io.BytesIO(buf.tobytes())
    mime = "image/jpeg" if format == "jpg" else "image/png"

    headers = {"X-Barcodes": json.dumps(found, ensure_ascii=False)}
    return StreamingResponse(bytes_io, media_type=mime, headers=headers)


@app.post("/backImage_json", response_model=BarcodeResponse)
async def backImage_json(
    file: UploadFile = File(...),
) -> Any:
    """
    Barcode -> JSON only (no image). Returns class, data, product, and bbox.
    """
    raw = await file.read()
    np_arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid image data"})

    H, W = img.shape[:2]

    barcodes = zbar_decode(img)
    detections: List[BarcodeDetection] = []

    if barcodes:
        for barcode in barcodes:
            data = (barcode.data.decode("utf-8")).strip()
            cls = barcode.type
            x, y, w, h = barcode.rect
            product = PRODUCTS_DB.get(data)

            detections.append(
                BarcodeDetection(
                    cls=cls,
                    data=data,
                    product=product,
                    bbox=BarcodeBox(x=x, y=y, w=w, h=h)
                )
            )

    return BarcodeResponse(
        status="success",
        image_width=W,
        image_height=H,
        count=len(detections),
        detections=detections
    )


# ---------------- Root ---------------- #
@app.get("/")
async def root():
    return {
        "app": APP_NAME,
        "model": DEFAULT_MODEL_PATH,
        "port": APP_PORT,
        "products_json_path": PRODUCTS_JSON_PATH,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=APP_PORT, reload=True)
