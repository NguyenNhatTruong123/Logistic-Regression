from cv2 import cv2
import numpy as np


img=cv2.imread('test19.png')
face_cascade = cv2.CascadeClassifier('C:/Users/truongnn/work_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/truongnn/work_env/Lib/site-packages/cv2/data/haarcascade_eye.xml')

#cv2.imshow('image',img)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
for (x, y, w, h) in faces: 
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    roi_gray = img_gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,200),1)
        cv2.putText(roi_color, 'Eye', (ex, ey-10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255), 1)  

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()