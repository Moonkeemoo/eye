"""
Легкий сервер живого дашборду навчання (без моделі/GPU).
Читає runs/detect/<run>/results.csv, який Ultralytics дописує щоепохи.

Запуск:
    source ~/cv/bin/activate
    cd /mnt/c/Users/tomoo/Documents/GitHub/eye
    uvicorn dashboard_server:app --app-dir web --host 0.0.0.0 --port 8000
"""
import csv
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent


app = FastAPI()


@app.get("/")
def dash():
    return HTMLResponse((ROOT / "dashboard.html").read_text(encoding="utf-8"))


@app.get("/progress")
def progress(run: str = "detect_v4m"):
    p = ROOT.parent / "runs" / "detect" / run / "results.csv"
    total = None
    ap = p.parent / "args.yaml"
    if ap.exists():
        for line in ap.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("epochs:"):
                try:
                    total = int(line.split(":")[1].strip())
                except ValueError:
                    pass
                break
    if not p.exists():
        return {"exists": False, "epochs": [], "series": {}, "total": total}
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
    return {"exists": True, "run": run, "epochs": epochs, "total": total,
            "series": {k: col(k) for k in keys}}
