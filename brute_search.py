import numpy as np
import cv2
import random
import sys

imgL = cv2.imread('conesL.PNG',0)
imgR = cv2.imread('conesR.png',0)

maxdisp = 16
np.set_printoptions(threshold=np.nan)
disps = imgL
print(imgL.shape)
heightL, widthL = imgL.shape
heightR, widthR = imgR.shape

i = 16
j = 16

for i in range(0, heightL):
    for j in range(0, widthL):
        disps[i, j] = 0

for i in range(25, heightL - 25):
    for j in range(25, widthL - 25):

        mindif = 255*20.0
        intens = 0.0

        for k in range(i - 5, i + 5):
            for l in range(j - 5, j + 5):
                intens = intens + imgL[k, l]

        for d in range(0, 16):
            intR = 0.0

            for k in range(i - 5, i + 5):
                for l in range(j - 5, j + 5):
                    intR = intR + imgR[k, l-d]


            if abs(intens*1.0 - intR) <= mindif:
                mindif = abs(intens*1.0 - intR)
                disps[i, j] = d*16.0

i = 25
j = 25




print(disps)

cv2.imshow('image',disps)
cv2.waitKey(0)
cv2.destroyAllWindows()