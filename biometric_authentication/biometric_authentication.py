import PySimpleGUI as sg
import iris_recognition
import face_recognition
import runpy

sg.theme('DarkAmber')
centroid = (0,0)
radius = 0
currentEye = 0
eyesList = []
numbersOfStarts = 0

greeting = 'Welcome to the "Biometrical Authentication" application.\
            \n"Biometrical Authentication" team are glad to know that\
            \nyou choose our software.'

layout = [[sg.Text(greeting)],
          [sg.Button('face recognition', key='face')],
          [sg.Button('fingerprint recognition', key='finger')],
          [sg.Button('iris recognition', key='iris')],
          [sg.Button('Exit',key = 'Exit')]]

window = sg.Window('Biometric Authentication',layout)

def main():
    while True:
        event,values = window.read()
        if(event == 'face'):
            runpy.run_module(mod_name='face_recognition.face_recognition')
        if(event == 'finger'):
            runpy.run_module(mod_name='fingerprint_recognition.fingerprint_recognition')
        if(event == 'iris'):
            runpy.run_module(mod_name='iris_recognition.iris_recognition')
        if(event == 'Exit'):
            break
    window.close()

if __name__ == "__main__":
	try:
		main()
	except:
		raise
