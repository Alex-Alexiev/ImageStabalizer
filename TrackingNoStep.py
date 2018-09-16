import numpy as np
import cv2
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
cv2.namedWindow("original")
cv2.namedWindow("smooth")

frameSize = 500
ratio = 0.75

def getCoords(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)    
    if len(faces) > 0:
        return faces[0]
    return [-1,0,0,0]

def getCoordsColour(img):

    blur = cv2.blur(img, (10,10))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 40, 0), (80, 255,255))
    conts, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:  
        area = cv2.contourArea(cnt)
        if (area > 2000):
            return  cv2.boundingRect(cnt)

    return -1,0,0,0


def getAngle(img1, img2):

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None) 
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    thetaTotal = 0
    n = 0
    deltaYThresh = 5

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
                thetaTotal += -math.atan(deltaX/deltaY)
            else:
                thetaTotal += math.atan(deltaX/deltaY)
            n+=1

    if (n>0):
        theta = thetaTotal/n
        theta = (theta*180)/math.pi
        return theta
    
    return 0


def rotate(img, angle):
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))

num = 0
firstLoop = True
k = 1
step = 2
angle = 0

try:    
    while(True):

        num+=1

        ret, img = cap.read()
        if (not ret):
            break

        imRaw = img.copy()

        # xo,yo,wo,ho = getCoords(img)

        # wi = np.size(img, 1)
        # hi = np.size(img, 0)

        # fx = frameSize
        # fy = ratio*frameSize

        # cx = int(xo+(wo/2))
        # cy = int(yo+(ho/2))

        #default to center
        if (firstLoop is True):
            # x = (wi-fx)/2
            # y = (hi-fy)/2
            prevImg = imRaw
            firstLoop = False
        elif (num > step):
            angle = k*getAngle(imRaw, prevImg)
            print(int(angle))
            num = 0

        # #if object was found
        # if (xo > -1):
        #     cv2.rectangle(img,(xo,yo),(xo+wo,yo+ho),(0,0,255),2)
        #     #cv2.circle(img,(cx, cy), 1, (0,0,255), -1)
        #     x = cx-(fx/2)
        #     y = cy-(fy/2)
        #     if (x < 0):
        #         x = 0
        #     if (x > (wi-fx)):
        #         x = wi-fx
        #     if (y < 0):
        #         y = 0
        #     if (y > (hi-fy)):
        #         y = hi-fy

       # if (angle is not None):
            #img = rotate(img, angle)
        # img = img[int(y):int(y+fy), int(x):int(x+fx)]
        
        cv2.imshow("smooth", img) 
        cv2.imshow("original", imRaw)
        
        if cv2.waitKey(33) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

except:
    pass