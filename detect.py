import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train5/weights/best.pt")
    model.predict(source="E:/Desktop/yolo/yolov8/data/images/val", imgsz=640, device="0", save=True)

# python detect.py
