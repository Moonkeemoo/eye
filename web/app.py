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
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "runs" / "detect"
MODEL = YOLO(str(ROOT.parent / "models" / "eye_iter2.pt"))
LOCK = threading.Lock()
DETLOG = deque(maxlen=6000)

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
    present = {}
    for d in dets:
        present[d["cls"]] = max(present.get(d["cls"], 0.0), d["conf"])
    DETLOG.append(present)
    return {"w": w, "h": h, "infer_ms": infer_ms, "dets": dets}


@app.get("/detstats")
def detstats(reset: int = 0):
    if reset:
        DETLOG.clear()
        return {"reset": True}
    frames = len(DETLOG)
    names = {0: "ceramic_mug", 1: "thermos", 2: "travel_mug"}
    out = {}
    for ci, nm in names.items():
        confs = [f[ci] for f in DETLOG if ci in f]
        out[nm] = {
            "frames_pct": round(100 * len(confs) / frames, 1) if frames else 0,
            "avg_conf": round(sum(confs) / len(confs), 3) if confs else 0,
            "max_conf": round(max(confs), 3) if confs else 0,
        }
    return {"frames": frames, "classes": out}


@app.post("/capture")
def capture(frame: UploadFile = File(...)):
    raw = frame.file.read()
    d = ROOT.parent / "data_raw" / "images"
    d.mkdir(parents=True, exist_ok=True)
    name = f"cap_{int(time.time() * 1000)}.jpg"
    (d / name).write_bytes(raw)
    return {"saved": name}


@app.post("/record")
def record(frame: UploadFile = File(...), cls: int = 2, conf: float = 0.25):
    raw = frame.file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "decode failed"}, status_code=400)
    cls = int(cls)
    h, w = img.shape[:2]
    with LOCK:
        r = MODEL.predict(img, conf=conf, imgsz=640, verbose=False)[0]
    names = {0: "ceramic_mug", 1: "thermos", 2: "travel_mug"}
    if len(r.boxes) == 0:
        return {"saved": False, "dets": []}
    best = max(r.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = [round(v, 1) for v in best.xyxy[0].tolist()]
    cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    d = ROOT.parent / "data_raw" / "images"
    d.mkdir(parents=True, exist_ok=True)
    nm = f"vid_{int(time.time() * 1000)}"
    cv2.imwrite(str(d / (nm + ".jpg")), img)
    with open(d / (nm + ".txt"), "w") as f:
        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return {"saved": True, "dets": [{"cls": cls, "name": names.get(cls, str(cls)),
            "conf": round(float(best.conf[0]), 3), "box": [x1, y1, x2, y2], "id": None}]}


TRAIN = {"proc": None, "name": None}


def _final_metrics(csvp):
    with open(csvp, encoding="utf-8") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [x for x in r if x and len(x) >= len(header)]
    if not rows:
        return None
    idx = {n: i for i, n in enumerate(header)}

    def val(row, k):
        try:
            return float(row[idx[k]]) if k in idx else None
        except ValueError:
            return None

    key = "metrics/mAP50-95(B)"
    best = rows[-1]
    for row in rows:
        v = val(row, key)
        if v is not None and (val(best, key) is None or v > val(best, key)):
            best = row

    def g(k):
        v = val(best, k)
        return round(v, 4) if v is not None else None

    return {"epochs": len(rows), "mAP50": g("metrics/mAP50(B)"), "mAP5095": g("metrics/mAP50-95(B)"),
            "P": g("metrics/precision(B)"), "R": g("metrics/recall(B)")}


@app.get("/runs")
def runs():
    out = []
    if RUNS.exists():
        for csvp in RUNS.glob("*/results.csv"):
            m = _final_metrics(csvp)
            if m:
                out.append({"name": csvp.parent.name, "mtime": csvp.stat().st_mtime, **m})
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return {"runs": out}


@app.post("/train")
def train_run(model: str = "m", epochs: int = 100):
    if model not in ("n", "s", "m"):
        model = "m"
    epochs = max(10, min(int(epochs), 600))
    if TRAIN["proc"] is not None and TRAIN["proc"].poll() is None:
        return {"error": f"вже триває: {TRAIN['name']}"}
    repo = ROOT.parent
    ds = repo / "datasets" / "eye-local"
    if ds.exists():
        shutil.rmtree(ds, ignore_errors=True)
    sp = subprocess.run([sys.executable, "split_dataset.py", "--src", "data_raw/images", "--out", "datasets/eye-local"],
                        cwd=str(repo), capture_output=True, text=True)
    if sp.returncode != 0:
        return {"error": "split: " + (sp.stderr or sp.stdout)[-300:]}
    (repo / "runs").mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in RUNS.glob("iter*")] if RUNS.exists() else []
    name = f"iter{len(existing) + 1}"
    logf = open(repo / "runs" / f"train_{name}.log", "w")
    proc = subprocess.Popen(
        ["yolo", "detect", "train", f"model=yolov8{model}.pt", "data=datasets/eye-local/data.yaml",
         f"epochs={epochs}", "imgsz=640", "batch=16", "device=0", f"name={name}"],
        cwd=str(repo), stdout=logf, stderr=subprocess.STDOUT)
    TRAIN["proc"] = proc
    TRAIN["name"] = name
    return {"started": name, "model": model, "epochs": epochs}
