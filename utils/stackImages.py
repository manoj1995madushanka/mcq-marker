import cv2
import numpy as np

# def stackImages(scale, imgArray, lables):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range(0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
#                                                 None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank] * rows
#         hor_con = [imageBlank] * rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor = np.hstack(imgArray)
#         ver = hor
#     return ver


def stackImagesNew(scale, imgArray, labels=None):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                img = imgArray[x][y]
                if img.shape[:2] == imgArray[0][0].shape[:2]:
                    img = cv2.resize(img, (0, 0), None, scale, scale)
                else:
                    img = cv2.resize(img, (int(width * scale), int(height * scale)), None)

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Add label if provided
                if labels and x < len(labels) and y < len(labels[x]):
                    label = labels[x][y]
                    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2, cv2.LINE_AA)

                imgArray[x][y] = img

        imageBlank = np.zeros_like(imgArray[0][0])
        hor = [np.hstack(imgArray[row]) for row in range(rows)]
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            img = imgArray[x]
            if img.shape[:2] == imgArray[0].shape[:2]:
                img = cv2.resize(img, (0, 0), None, scale, scale)
            else:
                img = cv2.resize(img, (int(width * scale), int(height * scale)), None)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Add label if provided
            if labels and x < len(labels):
                label = labels[x]
                cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)

            imgArray[x] = img
        ver = np.hstack(imgArray)
    return ver
