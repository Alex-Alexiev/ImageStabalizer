import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

cap = cv2.VideoCapture(0)
cv2.namedWindow("original")

first = True

totalAngle = 0

def getAngle(img1, img2):

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None) 
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    thetaTotal = 0
    n = 0

    for mat in matches[:10]:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        deltaX = x2-x1
        deltaY = y2-y1
        if (deltaY != 0):
            thetaTotal += -math.atan(deltaX/deltaY)
            n+=1

    if (n>0):
        theta = thetaTotal/n
        theta = (theta*180)/math.pi
        return theta
    
    return 0

num = 0

try:    
    while(True):
        num+=1
        ret, img = cap.read()
        if (not ret): 
            break
        if (not first and num > 3):  
            num = 0
            angle = getAngle(img, prevImg)
            totalAngle += angle
            print(int(angle))
            # int(addToAve(angle)), int(totalAngle))
        else:
            first = False
        cv2.imshow('original', img)
        prevImg = img 
        if cv2.waitKey(33) == 27:  
            break   

    cap.release()
    cv2.destroyAllWindows()

except:
    pass
