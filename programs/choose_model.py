from ultralytics import YOLO

detector_model = YOLO("models/detect/5epoch_detector.pt")

classifier_model = YOLO("models/classify/10epoch_v8m_cls.pt")