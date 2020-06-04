import cv2
import os
import sys
import numpy
from enhance import image_enhance
from skimage.morphology import skeletonize, thin
import PySimpleGUI as sg

sg.theme('DarkAmber')
centroid = (0,0)
radius = 0
currentEye = 0
eyesList = []
numbersOfStarts = 0

layout = [[sg.Text('Browse to a file')],
          [sg.Input(key='-FILE-', visible=False, enable_events=True), sg.FileBrowse()],
          [sg.MLine(size=(50,20),key = '-Output-')],
          [sg.Button('Recognition',key='recognition'), sg.Button('Exit',key = 'Exit')]]

window_with_finger = sg.Window('fingerprint_recognition',layout)

def removedot(invertThin):
    temp0 = numpy.array(invertThin[:])
    temp0 = numpy.array(temp0)
    temp1 = temp0/255
    temp2 = numpy.array(temp1)
    temp3 = numpy.array(temp2)

    enhanced_img = numpy.array(temp0)
    filter0 = numpy.zeros((10,10))
    W,H = temp0.shape[:2]
    filtersize = 6

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag +=1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag +=1
            if sum(filter0[0,:]) == 0:
                flag +=1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag +=1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))

    return temp2

def show_picture(name_of_window, image):
	cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(name_of_window, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def get_descriptors(img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)
	img = image_enhance.image_enhance(img)
	img = numpy.array(img, dtype=numpy.uint8)
	# Threshold
	ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	# Normalize to 0 and 1 range
	img[img == 255] = 1

	# Thinning
	skeleton = skeletonize(img)
	skeleton = numpy.array(skeleton, dtype=numpy.uint8)
	skeleton = removedot(skeleton)
	# Harris corners
	harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
	harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
	threshold_harris = 125
	# Extract keypoints
	keypoints = []
	for x in range(0, harris_normalized.shape[0]):
		for y in range(0, harris_normalized.shape[1]):
			if harris_normalized[x][y] > threshold_harris:
				keypoints.append(cv2.KeyPoint(y, x, 1))
	# Define descriptor
	orb = cv2.ORB_create()
	# Compute descriptors
	_, des = orb.compute(img, keypoints)
	#return (keypoints, des)
	return (des)

def parse_imgDB_to_csvDB():
	for i in range(1, 2):
		for j in range(1, 9):
			image_name = ("%s_%s" % (i, j))
			img = cv2.imread("PNG_database/" + image_name + ".png", cv2.IMREAD_GRAYSCALE)
			des = get_descriptors(img)
			numpy.savetxt("CSV_database/" + image_name + ".csv", des, delimiter=',')

def check_for_permission(path2img):
	img = cv2.imread(path2img, cv2.IMREAD_GRAYSCALE)
	des1 = get_descriptors(img)

	permission = "denied" 

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	for i in range(1, 17):
		for j in range(1, 9):
			image_name = ("%s_%s" % (i, j))
			des2 = numpy.loadtxt("fingerprint_recognition/CSV_database/%s.csv" % image_name, dtype=numpy.uint8, delimiter=',')

			matches = sorted(bf.match(des1, des2), key= lambda match:match.distance)

			score = 0
			for match in matches:
				score += match.distance

			window_with_finger['-Output-'].print(image_name + " " + str(100 - score/len(matches)))

			score_threshold = 23
			if score/len(matches) < score_threshold:
				permission = "allow"

	window_with_finger['-Output-'].print(permission)

def start_recognition():
	check_for_permission(values['Browse'])


while True:
	event,values = window_with_finger.read()
	if(event == 'recognition'):
		start_recognition()
	if(event == 'Exit'):
		break
	window_with_finger['-Output-'].update()
window_with_finger.close()