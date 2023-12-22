import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = 'Data/5'
counter = 0

while True:
    success, img = cap.read()
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

            # Determine the region to replace in imgWhite
            # h_offset = min(imgSize, imgResize.shape[0])
            # w_offset = min(imgSize, imgResize.shape[1])

            # Overlaying the cropped image onto imgWhite
            # imgWhite[0:h_offset, 0:w_offset] = imgResize[0:h_offset, 0:w_offset]
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            hGap = math.ceil((imgSize - hCal)/2)

            # Determine the region to replace in imgWhite
            # h_offset = min(imgSize, imgResize.shape[0])
            # w_offset = min(imgSize, imgResize.shape[1])

            # Overlaying the cropped image onto imgWhite
            # imgWhite[0:h_offset, 0:w_offset] = imgResize[0:h_offset, 0:w_offset]
            imgWhite[hGap:hCal+hGap, : ] = imgResize

            
        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    cv2.imshow("Image", img)

    # Check for the 'ESC' key (ASCII value 27)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

cap.release()
cv2.destroyAllWindows()
