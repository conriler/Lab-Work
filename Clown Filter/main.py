import cv2
from math import hypot
from djitellopy import Tello
import time
import keyboard
import JasFun

#Project goal: Put a clownface on a person using drone feed.



streamState = False
secondsToWait = 5
clownIMG = cv2.imread('clo.png')
cv2.imshow('Clown Image', clownIMG)
cv2.waitKey(0)
drone = Tello()
drone.connect()
print(drone.get_battery())
drone.streamoff()
face_cascade = cv2.CascadeClassifier('Frontal face.xml')
JasFun.droneCon()



while(streamState):
    #ret, frame = userVideo.read()
    frameClub = drone.get_frame_read()
    frame = frameClub.frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #neccesary for har casscade work
    userFace = face_cascade.detectMultiScale(gray, 1.3, 5)
    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor("shape_predictor_68_fa")

    final = cv2.imread('clo.png')
    for (x,y,w,h) in userFace:         #Some of this code was made with the help of a tutorial
        cv2.imshow('final', final)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('Drone Feed', frame)
        clownImgResized = cv2.resize(clownIMG,(w,h))
        cv2.imshow('Resized', clownImgResized)
        clownReGrey = cv2.cvtColor(clownImgResized, cv2.COLOR_BGR2GRAY)
        check, clownFilter = cv2.threshold(clownReGrey, 20, 255, cv2.THRESH_BINARY_INV)
        print(w/2)
        headArea = frame[y: y + h, x: x + w]
        cv2.imshow('f', headArea)
        bitwiseHead = cv2.bitwise_and(headArea, headArea, mask=clownFilter)
        final = cv2.add(bitwiseHead, clownImgResized)
        frame[y: y + h, x: x + w] = final


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

#droneVideo.release()






