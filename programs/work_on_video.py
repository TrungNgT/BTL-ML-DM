import os
import numpy as np
import cv2
from ultralytics.utils.plotting import Annotator

from choose_model import *


VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, "future_test/Virgin Atlantic - See the world differently.mp4")

video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W,_ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

threshold = 0.5

while ret:

    results = detector_model.predict(frame)[0]

    for result in results.boxes:
        x1, y1, x2, y2 = result.xyxy[0]

        h = int(y2 - y1)
        w = int(x2 - x1)
        x = int(x1)
        y = int(y1)
        crop_face = frame[y: y+h, x: x+w]

        emotion = classifier_model(crop_face)

        name_dict = emotion[0].names
        prob = emotion[0].probs

        label = name_dict[prob.top1]

        if threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
