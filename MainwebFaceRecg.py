import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

path = 'WebcamTest'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def attendance(name):
    file = open('p1.txt', 'r+')
    nameLst = file.readline()
    nameList = list(nameLst.split(','))
    if name not in nameList:
        file.write(f'{name},')

encodeListKnown = findEncodings(images)
print('Encoding Completed')

cap = cv2.VideoCapture(0)
i=0
while i<100:
    i = i +1
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].title()
            # print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)
            attendance(name)
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
file = open('p1.txt', 'r+')
nameLst = file.readline()
nameList = list(nameLst.split(','))
nameList.pop()
df = pd.read_csv('Attendence.csv', index_col=False)
di = list(df['Name'])
li = []
for i in range(len(di)):
    if di[i].title() in nameList:
        li.append(1)
    else:
        li.append(0)
now = datetime.now()
dt = now.strftime("%d/%m/%Y %H:%M:%S")
df[dt] = li
df.to_csv('Attendence.csv', index=False)
