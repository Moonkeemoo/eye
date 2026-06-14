"""
Веб-сервер живої детекції для проєкту eye.

Сторінку відкриваєш у браузері (на маку), кадри з вебки летять сюди,
тут модель eye_v1.pt рахується на 4070 Ti SUPER, назад — JSON з рамками.

Запуск (у WSL):
    source ~/cv/bin/activate
    cd /mnt/c/Users/tomoo/Documents/GitHub/eye
    uvicorn web.server:app --host 0.0.0.0 --port 8000

Назовні віддається через тимчасовий Cloudflare-тунель (HTTPS потрібен для камери).
"""
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT.parent / "models" / "eye_v1.pt"
model = YOLO(str(MODEL_PATH))

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/")
def index():
    return HTMLResponse((ROOT / "index.html").read_text(encoding="utf-8"))


@app.post("/infer")
async def infer(frame: UploadFile = File(...), conf: float = 0.35):
    raw = await frame.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "decode failed"}, status_code=400)
    h, w = img.shape[:2]
    t0 = time.perf_counter()
    r = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
    infer_ms = round((time.perf_counter() - t0) * 1000, 1)
    dets = []
    for b in r.boxes:
        ci = int(b.cls[0])
        x1, y1, x2, y2 = [round(v, 1) for v in b.xyxy[0].tolist()]
        dets.append(
            {
                "cls": ci,
                "name": r.names[ci],
                "conf": round(float(b.conf[0]), 3),
                "box": [x1, y1, x2, y2],
            }
        )
    return {"w": w, "h": h, "infer_ms": infer_ms, "dets": dets}
