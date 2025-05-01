import cv2


def showAnswers(img, myIndex, grading, answers, questions, choices):
    sectionWidth = int(img.shape[1] / questions)
    sectionHeight = int(img.shape[0] / choices)

    for x in range(0, questions):
        myAnswer = myIndex[x]
        # Finding center point
        cX = (myAnswer * sectionWidth) + sectionWidth // 2
        cY = (x * sectionHeight) + sectionHeight // 2

        if grading[x] == 1:
            myColor = (0, 255, 0)
        else:
            myColor = (0, 0, 255)
            correctAnswer = answers[x]
            cv2.circle(img,
                       ((correctAnswer * sectionWidth) + sectionWidth // 2, (x * sectionHeight) + sectionHeight // 2),
                       20, (0, 255, 0), cv2.FILLED)

        cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
    return img
