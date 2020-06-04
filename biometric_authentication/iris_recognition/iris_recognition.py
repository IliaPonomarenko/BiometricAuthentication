import cv2 as cv
import numpy as np
import os
import PySimpleGUI as sg

sg.theme('DarkAmber')
centroid = (0,0)
radius = 0
currentEye = 0
eyesList = []
numbersOfStarts = 0

layout = [[sg.Text('Browse to a file')],
          [sg.Input(key='-FILE-', visible=False, enable_events=True), sg.FileBrowse()],
          [sg.Text('For registry user')],
          [sg.Button('Registry',key = 'registry')],
          [sg.MLine(size=(50,20),key = '-Output-')],
          [sg.Button('Recognition',key='recognition'), sg.Button('Exit',key = 'Exit')]]

window = sg.Window('Window Title',layout)

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
        return(cropImg)
    return (resImg)



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
            break
    return(pupilImg)

def getPolar2CartImg(image, rad):
    height,width,channels = image.shape
    c = (float(height/2.0), float(width/2.0))
    res = cv.logPolar(image,c,50,cv.INTER_LINEAR+cv.WARP_FILL_OUTLIERS)
    res = res[int(res.shape[0]/2-(res.shape[0]/4)):int(res.shape[0]/2+(res.shape[0]/4)),int(res.shape[1]/2 + res.shape[1]/3.5):int(res.shape[1] - res.shape[1]/32)]
    lookUpTable = np.empty((1,256), np.uint8)
    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
           for c in range(res.shape[2]):
               res[y,x,c] = np.clip(1.1*res[y,x,c] + 30, 0, 255)


    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, 4) * 255.0, 0, 255)
    res = cv.LUT(res, lookUpTable)
    _,binaryImg = cv.threshold(res,70,255,cv.THRESH_BINARY)
    return (binaryImg)

def getIrisCode(binImg):
    height =int(binImg.shape[0])
    width = int(binImg.shape[1])
    resultat = np.empty((height,width), dtype= int)
    height = 0
    for y in binImg:
        width = 0
        buffer = 0
        for x in y:
            for z in x:
                if(z == 0):
                    buffer+=1
            if(buffer == 3):
                resultat[height,width] = 0.0
                buffer = 0
            else:
                resultat[height,width] = 1.0
                buffer = 0
            width+=1
        height+=1
    return(resultat)

def toFile(res,eye):
    name = eye.replace('.jpg','.txt')
    name = "iris_recognition/images/results/" + name
    np.savetxt(name,res, delimiter = ',')

def toFileRegistry(res,name):
    name = "iris_recognition/images/results/" + name + '.txt'
    np.savetxt(name,res, delimiter = ',')     

def toFileInputImg(res,name):
    name = name + ".txt"
    name = "iris_recognition/images/results/" + name
    np.savetxt(name,res, delimiter = ',')


def makeBD(image,eye):
    output = getPupil(image)
    iris = getIris(output)
    try:
       normImg = getPolar2CartImg(iris,radius)
    except Exception:
        window['-Output-'].print("Something going wrong because of preatretment image processing or scale")
        start()
    irisCode = getIrisCode(normImg)
    toFile(irisCode,eye)

def hemmingDif(str1,str2):
    difs = 0
    if(len(str1) != len(str2)):
        for ch1,ch2 in zip(str1,str2):
            if(ch1 != ch2):
                difs += 1
    return(difs)

def inputImage(image):
    output = getPupil(image)
    iris = getIris(output)
    try:
       normimg = getPolar2CartImg(iris,radius)
    except Exception:
        window['-Output-'].print("Something going wrong because of preatretment image processing or scale")
        start()

    irisCode = getIrisCode(normimg)

    path = "iris_recognition/images/results/"
    i = 0
    difference = 0
    owner = ""
    while(i<len(os.listdir(path))):
        currentPathToCode = os.listdir(path)[i]
        currentPathToCode = path + currentPathToCode
        currentCode = np.loadtxt(currentPathToCode,dtype= np.int, delimiter= ',')
        difference = diff(irisCode,currentCode)
        nameOfOwner = ""

        if(difference > 95):
            bufferForOwner = os.listdir(path)[i].replace('.txt', '')
            nameOfOwner = str(bufferForOwner)
            owner = nameOfOwner
        else:
            window['-Output-'].print("Nope" + " " + str(difference) + "%" + " "+ str(os.listdir(path)[i].replace(".txt", "")))
        i+=1
    if(owner != ""):
        window['-Output-'].print("Owner of eye is:" + owner)
        return(owner)
    else:
        window['-Output-'].print(" !!!!____Registry User___!!!!")
        owner = sg.popup_get_text('Registry','Input')
        toFileInputImg(irisCode,owner)

        


def diff(irisInput,irisDB):
    difs = 0
    elementsOfImage = 0
    for xI,xDB in zip(irisInput,irisDB):
        for chI,chDB in zip(xI,xDB):
            if(chI != chDB):
                difs +=1
            elementsOfImage +=1
    difs = 100 - (100*difs/elementsOfImage)
    difs = int(difs)
    return(difs)
    
           

def start():
    global numbersOfStarts
    numbersOfStarts +=1
    eyesList = os.listdir('iris_recognition/images/eyes')
    key = 0
    exitString = "exit"
    numberOfFiles = 0
    while True:
        if(numbersOfStarts <= 1):
          for i in os.listdir('iris_recognition/images/eyes'):
              eye = getNewEye(eyesList)
              frame = cv.imread("iris_recognition/images/eyes/" + eye)
              makeBD(frame,eye)
        path = values['Browse']
        image = cv.imread(path)
        owner = inputImage(image)
        if(owner != ""):
            break
def registryUser():
    path = sg.popup_get_file('Choose an image')
    name = sg.popup_get_text('Input Name')
    image = cv.imread(path)
    output = getPupil(image)
    iris = getIris(output)
    try:
       normImg = getPolar2CartImg(iris,radius)
    except Exception:
        window['-Output-'].print("Something going wrong because of preatretment image processing or scale")
        start()
    irisCode = getIrisCode(normImg)
    toFileInputImg(irisCode,name)

while True:
   event,values = window.read()
   if(event == 'registry'):
       registryUser()
   if(event == 'recognition'):
       start()
   if(event == 'Exit'):
       break
   window['-Output-'].update()
window.close()