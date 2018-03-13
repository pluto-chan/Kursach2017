from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

import cv2

np.set_printoptions(threshold=np.nan)

imgL = cv2.imread('conesL.PNG', 0)
imgR = cv2.imread('conesR.png', 0)

heightL, widthL = imgL.shape
heightR, widthR = imgR.shape

imgL3D = np.zeros((heightL, widthL, 3))
imgR3D = np.zeros((heightL, widthL, 3))

for k in range(0, 3):
    for i in range(0, heightL):
       for j in range(0, widthL):
            imgL3D[i, j, k] = 255 - imgL[i, j]
            imgR3D[i, j, k] = 255 - imgR[i, j]

segL = felzenszwalb(imgL3D, scale=300, sigma=0.5, min_size=100)
segR = felzenszwalb(imgR3D, scale=300, sigma=0.5, min_size=100)

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

ax[0, 0].imshow(mark_boundaries(imgL3D, segL))
ax[0, 1].imshow(mark_boundaries(imgR3D, segR))

plt.tight_layout()
plt.show()

TCol = 10
TGrad = 2

#colour absolute difference

def Ctadc(x, y, d):
    c = min(abs(imgL[y, x]*1.0 - imgR[y, x - d]*1.0), TCol)
    return  c

#computing horizontal and vertical gradients with Sobel operator first
from skimage.filters import sobel, sobel_h, sobel_v

sobelL_x = sobel_h(imgL)
sobelL_y = sobel_v(imgL)

sobelR_x = sobel_h(imgR)
sobelR_y = sobel_v(imgR)

#gradient absolute difference

def Ctadg(x, y, d):
    difx = abs(sobelL_x[y, x] * 1.0 - sobelR_x[y, x - d] * 1.0)
    dify = abs(sobelL_y[y, x] * 1.0 - sobelR_y[y, x - d] * 1.0)
    c = min (difx, TGrad) + min (dify, TGrad)
    return c

#Improved census transform

sobelL = sobel(imgL)
sobelR = sobel(imgR)

def ksiL(x, y, d, seg):
    w = 0
    meantens = 0.0
    meanx = 0.0
    meany = 0.0
    k = ""

    for i in range(0, heightL):
        for j in range(0, widthL):
            if segL[i, j] == seg:
                w += 1
                meantens += imgL[i, j]
                meanx += sobelL_x[i, j]
                meany += sobelL_y[i, j]

    #computing mean values for intensity and both gradients
    meantens /= w
    meanx /= w
    meany /= w

    for i in range(0, heightL):
        for j in range(0, widthL):
            if segL[i, j] == seg:
                if meantens <= imgL[i, j]:
                    k = k + "0"
                else:
                    k = k + "1"

    for i in range(0, heightL):
        for j in range(0, widthL):
            if segL[i, j] == seg:
                if meanx <= sobelL_x[i, j]:
                    k = k + "0"
                else:
                    k = k + "1"

    for i in range(0, heightL):
        for j in range(0, widthL):
            if segL[i, j] == seg:
                if meany <= sobelL_y[i, j]:
                    k = k + "0"
                else:
                    k = k + "1"

    print(k)
    return k


def ksiR(x, y, d, seg):
    w = 0
    meantens = 0.0
    meanx = 0.0
    meany = 0.0
    k = ""

    for i in range(0, heightL):
        for j in range(0, widthL):
            if segL[i, j] == seg:
                w += 1
                meantens += imgL[i, j]
                meanx += sobelL_x[i, j]
                meany += sobelL_y[i, j]

    # computing mean values for intensity and both gradients
    meantens /= w
    meanx /= w
    meany /= w

    for i in range(0, heightL):
        for j in range(0, widthL):
            if segL[i, j] == seg:
                if meantens <= imgL[i, j]:
                    k = k + "0"
                else:
                    k = k + "1"

    for i in range(0, heightL):
        for j in range(0, widthL):
            if segL[i, j] == seg:
                if meanx <= sobelL_x[i, j]:
                    k = k + "0"
                else:
                    k = k + "1"

    for i in range(0, heightL):
        for j in range(0, widthL):
            if segL[i, j] == seg:
                if meany <= sobelL_y[i, j]:
                    k = k + "0"
                else:
                    k = k + "1"

    print(k)
    return k


