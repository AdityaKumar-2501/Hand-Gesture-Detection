import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = 'Data/0'
counter = 0

classifier = Classifier("Model/keras_model.h5", "model/labels.txt")
labels = ['0', '1', '2','3','4','5']

np.random.seed(20)  # this stop randaom to create new random number generator after each iteration it just generates one time.
colorList = np.random.uniform(low =0,high=255, size = (len(labels), 3))


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # creating background image for croped image of hands
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        # Ensure the cropped region lies within the image boundaries
        imgCrop = img[max(0, y-offset):min(y+h+offset, img.shape[0]), max(0, x-offset):min(x+w+offset, img.shape[1])]

        # Overlaying the croped image onto the imgWhite
        # imgWhite[0:imgCrop.shape[0], 0:imgCrop.shape[1]] = imgCrop
 

        aspectRatio = h/w
        
        #  for height is greater than width
        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            wGap = math.ceil((imgSize - wCal)/2)

            imgWhite[:, wGap:wCal+wGap] = imgResize

            prediction,index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap, : ] = imgResize
            prediction,index = classifier.getPrediction(imgWhite , draw=False)
            print(prediction, index)
        
        labelColor = [int(c) for c in colorList[index]]

        # Draw a rectangular background
        cv2.rectangle(imgOutput, (x-offset, y - offset - 50), (x - offset+50, y-offset), labelColor, cv2.FILLED)
        # show character 
        cv2.putText(imgOutput, labels[index], (x-10,y-26), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 4)
        # Draw a rectangular Shape around the hand
        cv2.rectangle(imgOutput, (x-offset, y - offset), (x + w + offset, y+h+offset), labelColor, 2)
        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    cv2.imshow("Image", imgOutput)

    # Check for the 'ESC' key (ASCII value 27)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
