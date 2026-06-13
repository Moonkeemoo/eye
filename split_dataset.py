"""
Ділить ЛОКАЛЬНО розмічений датасет (labelImg, формат YOLO) на train/val/test
і генерує data.yaml для YOLOv8.

Вхід:  папка з фото (.jpg/.png) + поряд .txt-мітки + classes.txt.
Вихід: <out>/{images,labels}/{train,val,test} + <out>/data.yaml.

Запуск (у WSL; активуй будь-яке оточення з python, напр. ~/cv):
    source ~/cv/bin/activate
    cd /mnt/c/Users/tomoo/Documents/GitHub/eye
    python split_dataset.py --src data_raw/images --out datasets/eye-local
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="папка: фото + .txt мітки + classes.txt")
    ap.add_argument("--out", default="datasets/eye-local", help="куди скласти датасет")
    ap.add_argument("--ratios", default="0.7,0.2,0.1", help="train,val,test (сума=1)")
    ap.add_argument("--seed", type=int, default=42, help="фіксований сід -> відтворюваний split")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    train_r, val_r, _ = (float(x) for x in args.ratios.split(","))

    # 1. класи з classes.txt (порядок рядків = class_id)
    classes_file = src / "classes.txt"
    assert classes_file.exists(), f"Немає {classes_file} (labelImg створює його при розмітці)"
    names = [c.strip() for c in classes_file.read_text(encoding="utf-8").splitlines() if c.strip()]

    # 2. пари фото<->мітка (лишаємо лише фото, що мають .txt)
    images = [p for p in src.iterdir() if p.suffix.lower() in IMG_EXT]
    pairs = [(p, p.with_suffix(".txt")) for p in images if p.with_suffix(".txt").exists()]
    assert pairs, "Не знайдено жодної пари фото+мітка — спершу розміть у labelImg"

    # 3. перемішати з фіксованим сідом і поділити
    random.seed(args.seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train, n_val = int(n * train_r), int(n * val_r)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    # 4. розкласти по теках images/ та labels/
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy(img, out / "images" / split / img.name)
            shutil.copy(lbl, out / "labels" / split / lbl.name)

    # 5. data.yaml
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )

    print(f"Готово: {n} фото -> train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}")
    print(f"Класи ({len(names)}): {names}")
    print(f'У train.py постав:  DATA_YAML = "{(out / "data.yaml").as_posix()}"')


if __name__ == "__main__":
    main()
