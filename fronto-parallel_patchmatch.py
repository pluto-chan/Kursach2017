import numpy as np
import cv2
import random
import sys


gamma = 10
alpha = 0.9
TCol = 10
TGrad = 2
window = 10
maxdisp = 16

np.set_printoptions(threshold=np.nan)
imgL = cv2.imread('conesL.PNG', 0)
imgR = cv2.imread('conesR.png', 0)
sobelL = cv2.Sobel(imgL, cv2.CV_64F, 1, 0, ksize=5)
sobelR = cv2.Sobel(imgR, cv2.CV_64F, 1, 0, ksize=5)

disps = imgL
dispsR = imgR

heightL, widthL = imgL.shape
heightR, widthR = imgR.shape

heightL = int(heightL)
widthL = int(widthL)

for i in range(0, heightL):
    for j in range(0, widthL):
        disps[i, j] = random.uniform(0, maxdisp)
        dispsR[i, j] = random.uniform(0, maxdisp)

def cost (x, y):
    m = 0.0
    for qy in range (y - window, y + window + 1):
        for qx in range (x - window, x + window + 1):

            #weightning function
            dif = np.absolute(imgL[y, x] * 1.0 - imgL[qy, qx] * 1.0)
            colDif = float(3.0 * dif / gamma)
            w0 = np.exp(-colDif)

            #dissimilarity between matching pixels
            intensity_dif = min(3.0 * np.absolute(imgL[qy, qx] * 1.0 - imgR[qy, qx - disps[qy, qx]] * 1.0), TCol)

            gradient_dif = min(np.absolute(sobelL[qy, qx] * 1.0 - sobelR[qy, qx - disps[qy, qx]] * 1.0), TGrad)

            ro = (1.0 - alpha) * intensity_dif + alpha * gradient_dif

            #cost
            m = m + w0*ro

    return m

def cost_refine (x, y, ds):
    m = 0.0
    for qy in range (y - window, y + window + 1):
        for qx in range (x - window, x + window + 1):

            #weightning function
            dif = np.absolute(imgL[y, x] * 1.0 - imgL[qy, qx] * 1.0)
            colDif = float(3.0 * dif / gamma)
            w0 = np.exp(-colDif)

            #dissimilarity between matching pixels
            intensity_dif = min(3.0 * np.absolute(imgL[qy, qx] * 1.0 - imgR[qy, qx - int(ds)] * 1.0), TCol)

            gradient_dif = min(np.absolute(sobelL[qy, qx] * 1.0 - sobelR[qy, qx - int(ds)] * 1.0), TGrad)

            ro = (1.0 - alpha) * intensity_dif + alpha * gradient_dif

            #cost
            m = m + w0*ro

    return m

def costR (x, y):
    m = 0.0
    for qy in range (y - window, y + window + 1):
        for qx in range (x - window, x + window + 1):

            #weightning function
            dif = np.absolute(imgL[y, x] * 1.0 - imgL[qy, qx] * 1.0)
            colDif = float(3.0 * dif / gamma)
            w0 = np.exp(-colDif)

            #dissimilarity between matching pixels
            intensity_dif = min(3.0 * np.absolute(imgL[qy, qx] * 1.0 - imgR[qy, qx - dispsR[qy, qx - disps[y, x]]] * 1.0), TCol)

            gradient_dif = min(np.absolute(sobelL[qy, qx] * 1.0 - sobelR[qy, qx - dispsR[qy, qx - disps[y, x]]] * 1.0), TGrad)

            ro = (1.0 - alpha) * intensity_dif + alpha * gradient_dif

            #cost
            m = m + w0*ro

    return m

print (heightL, widthL)


for y in range(window + maxdisp, heightL/2 - window - 10):
    print ('y = ', y)
    for x in range(window + maxdisp, widthL/2 - window - 10):
        cv2.imwrite('result.png', disps)
        print ("pixel no ", x+y*widthL*1.0)
        #cost for the p-pixel
        current_cost = cost(x, y)

        # spatial propagation
        print ("spatial propagation step")
        for qx in range(x + 1, x + window + 1):
            q_cost = cost(qx, y)

            if q_cost < current_cost:
                current_cost = q_cost
                disps[y, x] = disps[y, qx]

        for qy in range(y+1, y + window + 1):
            for qx in range (x, x + window + 1):
                q_cost = cost(qx, qy)
                if q_cost < current_cost:
                    current_cost = q_cost
                    disps[y, x] = disps[qy, qx]

        #view propagation
        print ("view propagation step")
        right_cost = costR(x, y)
        if current_cost > right_cost:
            disps[y, x] = dispsR[y, x]

        #refinement
        print ("refinement step")
        deltamax = maxdisp/2

        while deltamax > 0:
            print("deltamax = ", deltamax )
            delta = int(random.uniform(-deltamax, deltamax))

            while (disps[y, x] + delta > 16) or (disps[y, x] + delta < 0):
                print("fail ds = ", disps[y, x] + delta)
                delta = int(random.uniform(-deltamax, deltamax))

            ds = disps[y, x] + delta
            ds_cost = cost_refine(x, y, ds)
            if ds_cost < current_cost:
                disps[y, x] = ds
            deltamax = deltamax / 2




for i in range(0, heightL):
    for j in range(0, widthL):
        disps[i, j] = 256/16.0*disps[i, j]
print(disps)
cv2.imshow('image',disps)
cv2.imwrite('result.png', disps)
cv2.waitKey(0)
cv2.destroyAllWindows()