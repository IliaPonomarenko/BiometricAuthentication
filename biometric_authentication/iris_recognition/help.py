import cv2 as cv
import numpy as np 

def draw_circles(storage,output):
    circles = np.asarray(storage)
    for circle in circles:
        Radius,x,y = int(circle[0][2]), int(circle[0][0]), int(circle[0][1])
        cv.circle(output,(x,y),1,cv.CV_RGB(0,255,0),-1,8,0)
        cv.circle(output,(x,y),Radius,cv,CV_RGB(255,0,0),3,8,0)

orig = cv.imread('images/R/S1001R02.jpg')
processed = cv.imread('images/R/S1001R02.jpg',cv.CV_LOAD_IMAGE_GRAYSCALE)
storage = new cv.Mat(orig.width,1,cv.CV_32FC3)
processed = cv.Canny(processed,5,70,3)
processed = cv.Smooth(processed,cv.CV_GAUSSIAN,7,7)

cv.HoughCircles(processed,storage,cv.HOUGH_GRADIENT,2,100.0,30,150,60,300)
draw_circles(storage,orig)

cv.namedWindow("original with circles")
cv.imshow("original with circles",orig)
cv.waitKey(0)