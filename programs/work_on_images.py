from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from ultralytics.utils.plotting import Annotator

detector_model = YOLO("train_process/train_wider_face_dataset_5epochs/weights/last.pt")

classifier_model = YOLO("train_process/classifier_model/train_with_FER/train20epoch/weights/train_with20epoches.pt")



image = cv2.imread("future_test/lecture-hall4.jpg")

results = detector_model.predict(image)

for r in results:
        
    annotator = Annotator(image)
        
    boxes = r.boxes
    for box in boxes:
            
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        
        h = int(b[3] - b[1])
        w = int(b[2] - b[0])
        x = int(b[0])
        y = int(b[1])
        crop_image = image[y: y+h, x: x+w]                          # crop the face from the source image
        emotion_class = classifier_model(crop_image)                # use the crop above as input for the classifier_model

        name_dict = emotion_class[0].names                      
        prob = emotion_class[0].probs

        
        print(prob)

        annotator.box_label(b, color=(160, 32, 240), txt_color=(255, 0 , 0))
          
    img = annotator.result()  
    cv2.imshow('YOLO V8 Detection', img)
    cv2.waitKey(0)
