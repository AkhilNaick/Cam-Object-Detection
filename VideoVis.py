import cv2
import numpy as np
from PIL import Image

webcam = cv2.VideoCapture(0)

k_size = 7
kernel = np.ones((5,5),np.uint8)
lower_color = np.array([30, 100, 50]) 
upper_color = np.array([90, 255, 255])

def findColor(source, lower, upper):
        ret, frame = source.read()

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frameHSV = cv2.blur(frameHSV, (k_size, k_size))
        frameHSV = cv2.morphologyEx(frameHSV, cv2.MORPH_CLOSE, kernel)

        mask = cv2.inRange(frameHSV, lower, upper)
        #res = cv2.bitwise_and(frame,frame, mask= mask)

        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()
        if bbox is not None: 
             x1, y1, x2, y2 = bbox

             frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)


        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
        #cv2.imshow('res', res)

while True: 
    findColor(webcam, lower_color, upper_color)
    
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
