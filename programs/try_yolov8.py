from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from ultralytics.utils.plotting import Annotator

detection_model = YOLO("yolov8n_100e.pt")

classifier_model = YOLO("C:/Users/MSI/runs/classify/train6/weights/train_with20epoches.pt")


#response = requests.get("https://images.unsplash.com/photo-1600880292203-757bb62b4baf?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80")

#image = Image.open(BytesIO(response.content))
#image = np.asarray(image)

image = cv2.imread("C:/Users/MSI/Desktop/c0273995-800px-wm.jpg")

results = detection_model.predict(image)
#print(results[0].boxes.data)

for r in results:
        
    annotator = Annotator(image)
        
    boxes = r.boxes
    for box in boxes:
            
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        #print(b)
        h = int(b[3] - b[1])
        w = int(b[2] - b[0])
        x = int(b[0])
        y = int(b[1])
        crop_image = image[y: y+h, x: x+w]                          # tôi cắt cái mặt ra khỏi ảnh gốc
        emotion_class = classifier_model(crop_image)                # xong cho đi vào cái model phân loại này

        name_dict = emotion_class[0].names
        prob = emotion_class[0].probs

        #print(name_dict)
        print(prob)
        
        #cv2.imshow('crop_face', crop_image)
        #cv2.waitKey(0)

        annotator.box_label(b, label=( name_dict[prob.top1] ), color=(160, 32, 240), txt_color=(255, 0 , 0))
          
    img = annotator.result()  
    cv2.imshow('YOLO V8 Detection', img)
    cv2.waitKey(0)

#plot_bboxes(image, results[0].boxes.data, conf=0.8)

#cv2.rectangle(image, results[0].boxes[1].data, (0, 0, 255), 2)

#cv2.imshow("output", image)
#cv2.waitKey(0)