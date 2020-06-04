
#Eye movement detection, Replacemenent of my patent ;)
# by Tanmoy Munshi
# Linkedin https://www.linkedin.com/in/tanmoymunshi/

import cv2
import numpy as np

cap = cv2.VideoCapture('eye2.mkv')
#cap = cv2.VideoCapture('eye.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret,frame = cap.read()   
    #roi = frame[5:160,170:1024]
    roi = frame[180:800,500:1200]

    gray_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi,(7,7),40)

    _,threshold = cv2.threshold(gray_roi,65,255,cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=lambda x: cv2.contourArea(x),reverse = True)

    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.drawContours(roi,[cnt],-1,(0,0,255),3)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),3)
        area = int((cv2.contourArea(cnt))/100)
        cv2.putText(roi,"area: "+str(area),(20,20),cv2.FONT_HERSHEY_COMPLEX,.7,(0, 255, 0), 2)
        
        
        break

    #cv2.imshow("frame",frame)
    cv2.imshow("threshold",threshold)
    cv2.imshow("roi",roi)
    cv2.imshow("gray_roi",gray_roi)


    key = cv2.waitKey(120)
    if key == 27:
        break

cv2.destroyAllWindows()