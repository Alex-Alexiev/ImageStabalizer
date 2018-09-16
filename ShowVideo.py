import cv2

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow("original")

try:    
    while(True):
        ret, img = cap.read()
        if (not ret):
            break
        blurry = cv2.Smooth(img, img, cv.CV_BLUR, 3)
        cv2.imshow('original', blurry)
        if cv2.waitKey(33) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

except:
    pass
