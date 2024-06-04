from ultralytics import YOLO

detector_model = YOLO("models/detect/10epoch_detector.pt")

classifier_model = YOLO("models/classify/20epoch_fer.pt")