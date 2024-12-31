import cv2
import numpy as np


def splitBoxes(img):
    rows = np.vsplit(img, 5)  # 5 is question count in the column
    cv2.imshow("Split", rows[0])

    boxes = []
    for row in rows:
        columns = np.hsplit(row,5) # 5 is the answer options (a,b,c,d,e)
        for box in columns:
            boxes.append(box)

    return boxes


