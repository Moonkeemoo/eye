"""
"Морда" — детекція з вебки в реальному часі.

ДЕ запускати: на МАКу (не на ПК). Мак зручний для real-time з камери,
а важке тренування вже зроблено на ПК.

Підготовка (один раз):
    pip install ultralytics opencv-python

Перенеси навчену модель з ПК сюди (поклади best.pt поруч із цим файлом):
    ПК: runs/detect/detect_v1/weights/best.pt  ->  AirDrop / хмара / scp

Запуск:
    python infer_webcam.py
Вихід — клавіша q.
"""

from ultralytics import YOLO
import cv2

model = YOLO("best.pt")          # навчена на ПК модель
cap = cv2.VideoCapture(0)        # 0 = вебка макбука

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # conf — поріг впевненості: показувати лише детекції >= 0.5
    results = model(frame, conf=0.5)

    cv2.imshow("eye — detection", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
