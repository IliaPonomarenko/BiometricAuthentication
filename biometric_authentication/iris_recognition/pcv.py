import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math 
import os

centroid = (0,0)
radius = 0
currentEye = 0
eyesList = []

def getNewEye(list):
    global currentEye
    if(currentEye >= len(list)):
        currentEye = 0
    newEye = list[currentEye]
    currentEye += 1 
    return (newEye)

def getIris(frame):
    iris = []
    copyImg = frame.copy()
    resImg = frame.copy()
    height,width,channels = frame.shape
    mask = np.zeros((int(height),int(width)),np.uint8)
    grayImg = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    grayImg = cv.Canny(grayImg,5,70,3)
    grayImg = cv.GaussianBlur(grayImg,(7,7),0)
    circles = getCircles(grayImg)
    for circle in circles:
        rad = int(circle[0][2]) - 30
        global radius
        radius = rad
        x = int(centroid[0] - rad)
        y = int(centroid[1] - rad)
        w = int(rad * 2)
        h = w
        cv.circle(mask,centroid,rad,(255,255,255),cv.FILLED)
        cv.bitwise_not(mask,mask)
        cv.subtract(frame,copyImg,resImg,mask)
        cropImg = resImg[y:y+h, x:x+w]
        buffer = getQuaterOfIris(cropImg).copy()
        del cropImg
        cropImg = buffer.copy()
        cv.imshow("res",cropImg)
        return(cropImg)
    return (resImg)

def getQuaterOfIris(image):
    h,w,_ = image.shape
    center = (h/2,w/2)
    y = int(center[0])
    x = int(center[1])
    y1 = int(y - y/2)
    y2 = int(y + y/2)
    quater = image[y1:y2, x:x*2 -1]
    quater = cv.GaussianBlur(quater,(5,5),0)
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, 1.5) * 255.0, 0, 255)
    res = cv.LUT(quater, lookUpTable)
    quater = res.copy()
    quater = cv.inRange(quater,(100,100,100),(190,190,190))
    contours,hierarchy = cv.findContours(quater,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv.contourArea)
    del quater
    quater = image[y1:y2, x:x*2-1].copy() 
    for contour in contours:
        moments = cv.moments(contour)
        area = moments['m00']
        if(area>50):
            iris = contour
            mask = np.zeros((quater.shape[:2]),np.uint8)
            mask = cv.drawContours(mask,iris,-1,(255,255,255),-1,cv.LINE_AA)
            mask = cv.fillConvexPoly(mask,iris,(255,255,255))
            buffer = np.zeros((quater.shape[:2]),np.uint8)
            buffer = cv.bitwise_and(quater,quater,mask=mask)
            quater = buffer.copy()
            break
    return (quater)

def getCircles(image):
    i = 80
    while i < 151:
        storage = np.zeros((image.shape[0],1))
        storage = cv.HoughCircles(image,cv.HOUGH_GRADIENT,2,100.0,30,i,100,140)
        circles = np.asarray(storage)
        if(len(circles) == 1):
            return circles
        i += 1
    return([])

def getPupil(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    channels = frame.shape[2]   
    pupilImg = np.zeros((height,width,1), np.uint8)
    pupilImg = cv.inRange(frame,(30,30,30),(80,80,80))
    contours,_ = cv.findContours(pupilImg,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv.contourArea)
    del pupilImg
    pupilImg = frame.copy()
    for contour in contours:
        moments = cv.moments(contour)
        area = moments['m00']
        if(area>50):
            pupilArea = area
            x = moments['m10']/area
            y = moments['m01']/area
            pupil = contour
            global centroid
            centroid = (int(x),int(y))
            mask = np.zeros((pupilImg.shape[0],pupilImg.shape[1],pupilImg.shape[2]),np.uint8)
            pupilImg = cv.fillConvexPoly(pupilImg,pupil,(0,0,0))
            cv.imshow("output",pupilImg)
            break
    return(pupilImg)

def getPolar2CartImg(image, rad):
    height,width,channels = image.shape
    value = np.sqrt(((image.shape[0]/2)**2.0)+((image.shape[1]/2)**2.0))
    imgRes = cv.linearPolar(image,(height/2,width/2),value,cv.WARP_POLAR_LINEAR + cv.WARP_POLAR_LOG)
    imgRes = imgRes.astype(np.uint8)
    return(imgRes)

cv.namedWindow("input")
cv.namedWindow("output")
cv.namedWindow("normalized")

eyesList = os.listdir('images/eyes')
key = 0
while True:
    eye = getNewEye(eyesList)
    frame = cv.imread("images/eyes/" + eye)
    iris = frame.copy()
    output = getPupil(frame)
    iris = getIris(output)
    cv.imshow("input",frame)
    cv.imshow("output",iris)
    normImg = iris.copy()
    normImg = getPolar2CartImg(iris,radius)
    cv.imshow("normalized",normImg)
    key = cv.waitKey(3000)
    if(key == 27 or key == 1048603):
        break
cv.destroyAllWindows()