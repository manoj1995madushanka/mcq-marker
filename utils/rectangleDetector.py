import cv2
import numpy as np


def detectRectangleContours(contours):
    ## check if it has 4 points
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        print("Area : ", area)
        if area > 50: # area that  we consider to check is it contour
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # getting corner count
            print("Corner points : ", len(approx))

            if len(approx) == 4:
                rectCon.append(i)
    print(rectCon)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon


def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx


# this function reorder the rectangle x,y points
def reOrder(mypoints):
    myPoints = mypoints.reshape((4, 2))
    mypointsNew = np.zeros((4, 1, 2), np.int32)

    add = myPoints.sum(1)
    mypointsNew[0] = myPoints[np.argmin(add)]      # [0,0]
    mypointsNew[3] = myPoints[np.argmax(add)]      # [w,h]

    diff = np.diff(myPoints, axis=1)
    mypointsNew[1] = myPoints[np.argmin(diff)]     # [w,0]
    mypointsNew[2] = myPoints[np.argmax(diff)]     # [0,h]

    return mypointsNew

