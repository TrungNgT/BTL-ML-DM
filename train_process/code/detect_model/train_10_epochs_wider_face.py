from ultralytics import YOLO
import faulthandler


if __name__ == "__main__":
    faulthandler.enable()  # start @ the beginning
    model = YOLO("yolov8n.pt")
    results = model.train(data="wider.yaml", epochs=10, batch=3, imgsz=640, device=[0], amp=False)