#chú ý các cài đặt làm theo hướng dẫn trong file txt gửi trên group mess.
#code này chỉ mang tính chất minh họa.

from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from ultralytics.utils.plotting import Annotator

model = YOLO("yolov8/yolov8n_100e.pt")

#response = requests.get("https://images.unsplash.com/photo-1600880292203-757bb62b4baf?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80")

#image = Image.open(BytesIO(response.content))
#image = np.asarray(image)

image = cv2.imread("test/many face.png")

results = model.predict(image)
print(results[0].boxes.data)

for r in results:
        
    annotator = Annotator(image)
        
    boxes = r.boxes
    for box in boxes:
            
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])
          
    img = annotator.result()  
    cv2.imshow('YOLO V8 Detection', img)
    cv2.waitKey(0)

#plot_bboxes(image, results[0].boxes.data, conf=0.8)

#cv2.rectangle(image, results[0].boxes[1].data, (0, 0, 255), 2)

#cv2.imshow("output", image)
#cv2.waitKey(0)