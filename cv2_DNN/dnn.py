import cv2
import numpy as np
#from google.colab.patches import cv2_imshow #To run in Google Colab, uncomment this line

# Load the DNN Face Detector model
face_detector = cv2.dnn.readNetFromCaffe("cv2_DNN/deploy.prototxt.txt", "cv2_DNN/res10_300x300_ssd_iter_140000.caffemodel")

# Read the input image
print("Read the input image")
image = cv2.imread("test/many face.png")

# Get the height and width of the input image
(h, w) = image.shape[:2]

# Preprocess the image by resizing it and converting it to a blob
print("Preprocessing the image")
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Feed the blob as input to the DNN Face Detector model
face_detector.setInput(blob)
detections = face_detector.forward()

# Loop over the detections and draw a rectangle around each face
print("Iterate over detections and create a rectangle")
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections
    if confidence > 0.5:
        # Get the bounding box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw a rectangle around the face
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

# Show the output image
print("Show the final output")
cv2.imshow("Output", image) #To run in Google Colab, comment out this line Colab notebook
#cv2_imshow(image) #To run in Google Colab, uncomment this line
cv2.waitKey(0)
print("Completed")