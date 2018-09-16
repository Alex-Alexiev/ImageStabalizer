import numpy as mp
import cv2

face_cascade = cv2.CascadeClassifier('C:/Dev/LanguagesLibraries/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

try:
    while(True):
        ret, img = cap.read()

        if (not ret):
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        print(faces[0])

        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('video', img)

        if cv2.waitKey(33) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

except:
    pass
