"""
"Морда" — детекція з вебки в реальному часі. Запускати на МАКу.

Модель уже лежить у репозиторії (models/eye_v1.pt), тож на маку досить:
    git clone git@github.com:Moonkeemoo/eye.git     # або `git pull`, якщо вже клоновано
    cd eye
    pip install ultralytics opencv-python
    python infer_webcam.py
Вихід — клавіша q.

Якщо macOS не дає доступ до камери: System Settings → Privacy & Security →
Camera → дозволь Терміналу (або iTerm/VS Code), з якого запускаєш.
"""

from ultralytics import YOLO
import cv2

model = YOLO("models/eye_v1.pt")   # навчена на ПК модель (ітерація 1)
cap = cv2.VideoCapture(0)          # 0 = вебка макбука

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
