from utils.DisplayAnswers import showAnswers
from utils.SplitImage import splitBoxes
from utils.rectangleDetector import detectRectangleContours, getCornerPoints, reOrder
from utils.stackImages import stackImagesNew
import cv2
import numpy as np

path = "img.png"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
answers = [1, 2, 0, 1, 4]

img = cv2.imread(path)

# resize image
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContours = img.copy()
imgWarpColored = None  # Only contains question answers contour
imgThresh = None

# Preprocessing
# convert to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# Find contours
countours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# -1 means index and that value represent all, 10 is the thickness
cv2.drawContours(imgContours, countours, -1, (0, 255, 0), 10)
# Find rectangle contours
rectangleContours = detectRectangleContours(countours)

if len(rectangleContours) >= 2:
    biggestContour = getCornerPoints(rectangleContours[0])  # because it is reverse ordered and this is answers
    gradePoints = getCornerPoints(rectangleContours[1])  # this is getting grade rectangle points of the image
    print("BIGGEST CONTOUR")
    print(biggestContour.shape)  # this is printing x,y coordinates of the 4 corner points of biggest rectangle

    if biggestContour is not None and gradePoints is not None and biggestContour.size != 0 and gradePoints.size != 0:
        cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
        cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

        biggestContour = reOrder(biggestContour)
        gradePoints = reOrder(gradePoints)

        # Multiple answer section
        point1 = np.float32(biggestContour)
        point2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

        matrix = cv2.getPerspectiveTransform(point1, point2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # Grade point section
        gPoint1 = np.float32(gradePoints)
        gPoint2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])

        matrixG = cv2.getPerspectiveTransform(gPoint1, gPoint2)
        imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
        #cv2.imshow("Grade", imgGradeDisplay)

        # Apply Threshold to identify marks easilly
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

        boxes = splitBoxes(imgThresh)
        # cv2.imshow("Test", boxes[2])
        # print(cv2.countNonZero(boxes[1]), cv2.countNonZero(boxes[2]))

        # Getting non zero pixel value for each box
        myPixelVal = np.zeros((questions, choices))
        countColumn = 0
        countRow = 0
        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countRow][countColumn] = totalPixels
            countColumn += 1
            if countColumn == choices:
                countRow += 1
                countColumn = 0

        # print("PIXELS")
        # print(myPixelVal)

        # Finding student selected choices
        myIndex = []
        for x in range(0, questions):
            arr = myPixelVal[x]
            myIndexVal = np.where(arr == np.amax(arr))
            myIndex.append(myIndexVal[0][0])
        # print(myIndex)

        # Grading
        grading = []
        for x in range(0, questions):
            if answers[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)
        print("Printing Grading : ", grading)

        # Score
        score = (sum(grading) / questions) * 100
        print("Score :", score)

        # Displaying answers
        imgResult = imgWarpColored.copy()
        imgResult = showAnswers(imgResult, myIndex, grading, answers, questions, choices)

        imgRawDrawing = np.zeros_like(imgWarpColored)
        imgRawDrawing = showAnswers(imgRawDrawing, myIndex, grading, answers, questions, choices)

        inverseMatrix = cv2.getPerspectiveTransform(point2, point1)
        imgInverseWarp = cv2.warpPerspective(imgRawDrawing, inverseMatrix, (widthImg, heightImg))

        imgRawGrade = np.zeros_like(imgGradeDisplay)
        cv2.putText(imgRawGrade, str(int(score)) + "%", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
        #cv2.imshow("Grade",imgRawGrade)

        inverseMatrixG = cv2.getPerspectiveTransform(gPoint2, gPoint1)
        imgInverseGradeDisplay = cv2.warpPerspective(imgRawGrade, inverseMatrixG, (widthImg, heightImg))

        imgFinal = cv2.addWeighted(imgFinal, 1, imgInverseWarp, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInverseGradeDisplay, 1, 0)

    else:
        print("Corner points are empty, cannot proceed.")

    imgBlank = np.zeros_like(img)  # create blank image
    imgArray = (
        [img, imgGray, imgBlur, imgCanny],
        [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
        [imgResult, imgRawDrawing, imgInverseWarp, imgFinal]
    )

    labels = [
        ["Original", "Gray", "Blur", "Canny"],
        ["Contour", "Biggest Contour", "Warp", "Threshold"],
        ["Result", "Raw Drawing", "Inverse Warp", "Final"]
    ]
    imgStacked = stackImagesNew(0.3, imgArray, labels)

    cv2.imshow("Stacked Images", imgStacked)
    cv2.imshow("Final Image", imgFinal)
    cv2.waitKey(0)
else:
    print("Not enough rectangle contours detected")