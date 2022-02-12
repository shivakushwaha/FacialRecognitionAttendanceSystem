import cv2
import numpy as np
import face_recognition
# Fist step is to loading the images from resources folder
imgShiva = face_recognition.load_image_file('resources/shiva.jpg')
imgShiva = cv2.cvtColor(imgShiva,cv2.COLOR_BGR2RGB)

imgShivatest1 = face_recognition.load_image_file('resources/shivatest.jpg')
imgShivatest1 = cv2.cvtColor(imgShivatest1,cv2.COLOR_BGR2RGB)

imgShivatest = face_recognition.load_image_file('resources/Ajmaltest.jpg')
imgShivatest = cv2.cvtColor(imgShivatest,cv2.COLOR_BGR2RGB)
# 1
faceloc = face_recognition.face_locations(imgShiva)[0]
encodeShiva = face_recognition.face_encodings(imgShiva)[0]
cv2.rectangle(imgShiva,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,0,255),2)
# 2
faceloctest1 = face_recognition.face_locations(imgShivatest1)[0]
encodeShivatest1 = face_recognition.face_encodings(imgShivatest1)[0]
cv2.rectangle(imgShivatest1,(faceloctest1[3],faceloctest1[0]),(faceloctest1[1],faceloctest1[2]),(255,0,0),2)
# 3
faceloctest = face_recognition.face_locations(imgShivatest)[0]
encodeShivatest = face_recognition.face_encodings(imgShivatest)[0]
cv2.rectangle(imgShivatest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,0),2)

result = face_recognition.compare_faces([encodeShiva],encodeShivatest)
result1 = face_recognition.compare_faces([encodeShiva],encodeShivatest1)

facedis = face_recognition.face_distance([encodeShiva],encodeShivatest)
facedis1 = face_recognition.face_distance([encodeShiva],encodeShivatest1)

print(result ,facedis)
print((result ,facedis1))
cv2.putText(imgShivatest,f'{result} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.putText(imgShivatest1,f'{result1} {round(facedis1[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Ajmal image',imgShivatest)
cv2.imshow('shiva image',imgShiva)
cv2.imshow('shiva image1',imgShivatest1)
cv2.waitKey(0)

