from cv2 import cv2

im1=cv2.imread('image1.png')

grayImage = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 128, 255, cv2.THRESH_BINARY)
 
cv2.imshow('Black white image', blackAndWhiteImage)
