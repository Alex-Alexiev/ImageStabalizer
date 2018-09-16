import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

img1 = cv2.imread('ori2.jpg', 0)
img2 = cv2.imread('ori2.jpg', 0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

list_kp1 = []
list_kp2 = []
deltaY = []
deltaX = []
theta = []
thetaTotal = 0
n = 0

# For each match...
for mat in matches:
    n += 1
    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt
    
    deltaX = x2-x1
    deltaY = y2-y1

    thetaTotal += math.atan(deltaX/deltaY)

    #theta.append(-math.atan(deltaX/deltaY))
    # deltaX.append(int(x2-x1))
    # deltaY.append(int(y2-y1))

    # # Append to each list
    # list_kp1.append((int(x1), int(y1)))
    # list_kp2.append((int(x2), int(y2)))

theta = thetaTotal/n
theta = (theta*180)/math.pi



# Draw first 10 matches.
# print(list_kp1, list_kp2)
# plt.plot(list_kp1)
# plt.show()

print(theta)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1500,500)
cv2.imshow('image', img3)

while (True):
     if cv2.waitKey(33) == 27:
            break

