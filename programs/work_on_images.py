import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from ultralytics.utils.plotting import Annotator
from choose_model import *

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

        annotator.box_label(b, label=name_dict[prob.top1], color=(102, 255, 102), txt_color=(255, 0 , 0))
          
    img = annotator.result()  
    cv2.imshow('YOLO V8 Detection', img)
    cv2.waitKey(0)
