import numpy as np
import cv2 as cv
import time
face_cascade = cv.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('../data/haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv.CascadeClassifier('../data/haarcascade_smile.xml')

cap = cv.VideoCapture(0)

#cap = cv.VideoCapture('../data/Megamind.avi')

time.sleep(1) ### letting the camera autofocus

while(True):

    ret, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes[:2]:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smile = smile_cascade.detectMultiScale( roi_gray, scaleFactor=1.2, minNeighbors=20)
        for (sx,sy,sw,sh) in smile[:2]:
            cv.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)


    cv.imshow('img',img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.waitKey(0)
cv.destroyAllWindows()