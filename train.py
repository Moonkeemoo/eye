"""
Тренування YOLOv8 на власному датасеті: mug / thermos / glass.

ДЕ запускати: у WSL Ubuntu-24.04, попередньо активувавши середовище:
    wsl -d Ubuntu-24.04
    source ~/cv/bin/activate
    python train.py

Перед першим запуском треба підставити свій датасет із Roboflow
(Export -> YOLOv8). Покрокова інструкція: docs/roboflow-guide.md.

Результат ляже в: runs/detect/detect_v1/weights/best.pt
"""

from ultralytics import YOLO
# from roboflow import Roboflow   # розкоментуй, якщо качаєш датасет напряму з Roboflow


# =============================================================
# 1. ДАТАСЕТ
# =============================================================
# Варіант А — завантажити з Roboflow (встав значення зі сніпета Export):
#
#   rf = Roboflow(api_key="ВСТАВ_СВІЙ_КЛЮЧ")
#   project = rf.workspace("ВСТАВ_WORKSPACE").project("ВСТАВ_PROJECT")
#   dataset = project.version(1).download("yolov8")
#   DATA_YAML = f"{dataset.location}/data.yaml"
#
# Варіант Б — датасет уже розпакований локально: вкажи шлях до data.yaml.
DATA_YAML = "datasets/eye-1/data.yaml"   # <- підправ під реальний шлях


# =============================================================
# 2. МОДЕЛЬ (transfer learning)
# =============================================================
# yolov8n.pt — передтренована на COCO (мільйони фото). Вона вже вміє бачити
# краї/текстури/форми; ми лише доналаштовуємо її під наші 3 класи.
# Тому й вистачає сотень фото, а не мільйонів.
model = YOLO("yolov8n.pt")   # n = nano (найлегша й найшвидша)


# =============================================================
# 3. ТРЕНУВАННЯ
# =============================================================
results = model.train(
    data=DATA_YAML,
    epochs=50,         # скільки разів пройти весь датасет (мало -> недонавчання, багато -> overfitting)
    imgsz=640,         # усе ресайзиться до 640x640
    batch=16,          # фото за раз; 4070 Ti SUPER (16 ГБ) потягне й більше
    patience=10,       # рання зупинка: стоп, якщо val не кращає 10 епох (захист від overfitting)
    device=0,          # GPU 0 = RTX 4070 Ti SUPER
    name="detect_v1",  # папка результатів: runs/detect/detect_v1
)

print("Готово. Найкращі ваги -> runs/detect/detect_v1/weights/best.pt")
print("Метрики й графіки -> runs/detect/detect_v1/")
