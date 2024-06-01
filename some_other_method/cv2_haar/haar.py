import cv2

face_cascade = cv2.CascadeClassifier("cv2_haar/haarcascade_frontalface_default.xml")            # truyền vào pretrained model

image = cv2.imread("test/many face.png")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

faces = face_cascade.detectMultiScale(gray_image, 1.1, 10)

for (x, y, width, height) in faces:
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)

cv2.imshow("Detected faces", image)

cv2.waitKey()

for f in faces:
    print(f)