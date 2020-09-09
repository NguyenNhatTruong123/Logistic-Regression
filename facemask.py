from cv2 import cv2
import numpy as np
from imageai.Detection import ObjectDetection
import os
from PIL import Image

execution_path = os.getcwd()
img=cv2.imread('test20.png')
face_cascade = cv2.CascadeClassifier('C:/Users/truongnn/work_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
for (x, y, w, h) in faces: 
    cv2.rectangle(img,(x,y+int(round(h/2))),(x+w,y+h), (0, 0, 255), 1)
    mask17=img[y+int(round(h/2)):y+h, x:x+w]
    Image.fromarray(mask17).save(os.path.join(execution_path, 'maskdt.png'))


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path,'maskdt.png'), output_image_path=os.path.join(execution_path,'masknew.png'))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )


cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()