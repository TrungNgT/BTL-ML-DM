import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the YOLOv8 model
detect_model = YOLO('models/pretrained_model/yolov8n_100e.pt')

emotion_classifier = YOLO('train_process/classifier_model/train_with_FETD/train15epoch/weights/last.pt')

# Open the video file
# video_path = "choreography.mp4"
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = detect_model(frame)[0]

        annotated_frame = Annotator(frame)

        # Nếu có GPU thì ông thêm vào cái device, máy tôi để là mps, GPU NVIDIA thì phải điền khác
        # results = model(frame, device="mps")

        # Visualize the results on the frame
        #annotated_frame = results[0].plot()

        for res in results.boxes:
            x1, y1, x2, y2 = res.xyxy[0]

            #b = result.xyxy[0]
            #print(b)

            h = int(y2 - y1)
            w = int(x2 - x1)
            x = int(x1)
            y = int(y1)
            crop_face = frame[y: y+h, x: x+w]

            emotion = emotion_classifier(crop_face)

            name_dict = emotion[0].names
            prob = emotion[0].probs

            label = name_dict[prob.top1]

            annotated_frame.box_label([x1, y1, x2, y2], label)
            

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame.result())

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()