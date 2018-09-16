import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow("original")
firstLoop = True

def getCoords(img2):

    blur = cv2.blur(img2, (10,10))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 40, 0), (80, 255,255))
    conts, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       
    for cnt in contours:  
        area = cv2.contourArea(cnt)
        if (area > 2000):
            return  cv2.boundingRect(cnt)

    return -1,0,0,0

try:    
    while(True):

        ret, img = cap.read()

        if (not ret):
            break

        xo,yo,wo,ho = getCoords(img)

        wi = np.size(img, 1)
        hi = np.size(img, 0)

        frameSize = 500
        ratio = 0.75
        fx = frameSize
        fy = ratio*frameSize

        cx = int(xo+(wo/2))
        cy = int(yo+(ho/2))

        #default to center
        if (firstLoop is True):
            x = (wi-fx)/2
            y = (hi-fy)/2
            firstLoop = False

        if (xo > -1):
            cv2.rectangle(img,(xo,yo),(xo+wo,yo+ho),(0,0,255),2)
            cv2.circle(img,(cx, cy), 1, (0,0,255), -1)
            #cv2.drawContours(img, contours, -1, (0,255,0), 3)
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

        print(y,y+fy, x, x+fx)
        img = img[int(y):int(y+fy), int(x):int(x+fx)]
        #img = img[90:390, 120:520]
        cv2.imshow('original', img)
        
        if cv2.waitKey(33) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

except:
    pass
