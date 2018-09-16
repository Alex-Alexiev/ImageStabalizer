import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cap = cv2.VideoCapture(1)

cv2.namedWindow("original")
cv2.namedWindow("smooth")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

rough = cv2.VideoWriter('rough.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))
smooth = cv2.VideoWriter('nice.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

def getCoords(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)    
    if len(faces) > 0:
        return faces[0]
    return [-1,0,0,0]

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

    ccw = 3
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

frameSize = 400
ratio = 0.75

step = 3

try:    
    while(True):
        num+=1
        ret, img = cap.read()
        if (not ret): 
            break
        imRaw = img.copy()

        xo,yo,wo,ho = getCoords(img)

        wi = np.size(img, 1)
        hi = np.size(img, 0)

        fx = frameSize
        fy = ratio*frameSize

        cx = int(xo+(wo/2))
        cy = int(yo+(ho/2))

        if (first):
            x = (wi-fx)/2
            y = (hi-fy)/2
            first = False
            prevImg = img.copy()
        
        if (xo > -1):
            cv2.rectangle(img,(xo,yo),(xo+wo,yo+ho),(0,0,255),2)
            #cv2.circle(img,(cx, cy), 1, (0,0,255), -1)
            x = cx-(fx/2)
            y = cy-(fy/2)
            if (x < 0):
                x = 0
            if (x > (wi-fx)):
                x = wi-fx
            if (y < 0):
                y = 0
            if (y > (hi-fy)):
                y = hi-fy

        if (num > step):  
            num = 0 
            # angV = getAngle(imRaw, prevImg)
            # angle += 0.09*angV
            # print(int(angle), int(angV))
            prevImg = imRaw  

        img = img[int(y):int(y+fy), int(x):int(x+fx)]
       # img = rotate(img, angle)
        cv2.imshow('smooth', img)
        smooth.write(img)
        cv2.imshow('original', imRaw)
        rough.write(imRaw)
        if cv2.waitKey(33) == 27:  
            break    

    cap.release()  
    rough.release()
    smooth.release()
    cv2.destroyAllWindows()

except:
    pass
