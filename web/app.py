"""
Єдиний сервер проєкту eye:
  /          → дашборд навчання (графіки)
  /camera    → жива камера з трекінгом (ID об'єктів) + дані розпізнавання
  /progress  → метрики навчання з results.csv (для дашборду)
  /track     → приймає кадр, проганяє model.track (persist) на GPU, віддає рамки + track id

Запуск:
    source ~/cv/bin/activate
    cd /mnt/c/Users/tomoo/Documents/GitHub/eye
    uvicorn app:app --app-dir web --host 0.0.0.0 --port 8000
"""
import csv
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "runs" / "detect"
MODEL = YOLO(str(ROOT.parent / "models" / "eye_v4m.pt"))
LOCK = threading.Lock()

app = FastAPI()


def page(name):
    return HTMLResponse((ROOT / name).read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/")
def home():
    return page("dashboard.html")


@app.get("/camera")
def camera():
    return page("camera.html")


@app.get("/progress")
def progress(run: str = ""):
    if not run:
        cands = list(RUNS.glob("*/results.csv"))
        if cands:
            run = max(cands, key=lambda p: p.stat().st_mtime).parent.name
    p = (RUNS / run / "results.csv") if run else None
    total = None
    if p is not None:
        ap = p.parent / "args.yaml"
        if ap.exists():
            for line in ap.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("epochs:"):
                    try:
                        total = int(line.split(":")[1].strip())
                    except ValueError:
                        pass
                    break
    if p is None or not p.exists():
        return {"exists": False, "epochs": [], "series": {}, "total": total, "run": run}
    with open(p, encoding="utf-8") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        data = [row for row in r if row and len(row) >= len(header)]
    idx = {n: i for i, n in enumerate(header)}

    def col(name):
        if name not in idx:
            return []
        out = []
        for row in data:
            try:
                out.append(float(row[idx[name]]))
            except ValueError:
                out.append(None)
        return out

    epochs = [int(x) for x in col("epoch")]
    keys = [
        "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
        "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    ]
    idle_s = round(time.time() - p.stat().st_mtime, 1)
    return {"exists": True, "run": run, "epochs": epochs, "total": total,
            "idle_s": idle_s, "series": {k: col(k) for k in keys}}


@app.post("/track")
def track(frame: UploadFile = File(...), conf: float = 0.35):
    raw = frame.file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "decode failed"}, status_code=400)
    h, w = img.shape[:2]
    t0 = time.perf_counter()
    with LOCK:
        r = MODEL.track(img, persist=True, conf=conf, imgsz=640, verbose=False)[0]
    infer_ms = round((time.perf_counter() - t0) * 1000, 1)
    dets = []
    for b in r.boxes:
        ci = int(b.cls[0])
        x1, y1, x2, y2 = [round(v, 1) for v in b.xyxy[0].tolist()]
        tid = int(b.id[0]) if b.id is not None else None
        dets.append({"cls": ci, "name": r.names[ci], "conf": round(float(b.conf[0]), 3),
                     "box": [x1, y1, x2, y2], "id": tid})
    return {"w": w, "h": h, "infer_ms": infer_ms, "dets": dets}
