import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

cap = cv2.VideoCapture(1)
cv2.namedWindow("original")
#cv2.namedWindow("smooth")

first = True

totalAngle = 0

def rotate(img, angle):
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))

def getAngle(img1, img2):

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None) 
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    thetaTotal = 0
    n = 0
    deltaYThresh = 3

    ccw = 0.5
    cw = 1

    for mat in matches[:15]:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        
        deltaX = x2-x1
        deltaY = y2-y1 
        if (deltaY < deltaYThresh):
            deltaY = 0
        if (deltaY != 0):
            if (deltaY < 0):
                thetaTotal += -math.atan(deltaX/deltaY)*ccw
            else:
                thetaTotal += math.atan(deltaX/deltaY)*cw
            n+=1

    if (n>0):
        theta = thetaTotal/n
        theta = (theta*180)/math.pi
        return theta
    
    return 0

num = 0
first = True
angle = 0

try:    
    while(True):
        num+=1
        ret, img = cap.read()
        if (not ret): 
            break
        imRaw = img.copy()
        if (first):
            first = False
            prevImg = img
        if (num > 3):  
            num = 0 
            angV = getAngle(imRaw, prevImg)
            angle += 0.09*angV
            print(int(angle), int(angV))
            prevImg = imRaw  
        img = rotate(img, angle)
        cv2.imshow('smooth', img)
        #cv2.imshow('original', imRaw)
        if cv2.waitKey(33) == 27:  
            break    

    cap.release()
    cv2.destroyAllWindows()

except:
    pass
