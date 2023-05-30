import socket
import cv2
import numpy as np
import time
from custom_socket import CustomSocket
import json


l=[]
for c in range(10):
    try:
        ret, frame = cv2.VideoCapture(c).read()

        if ret: 
            l.append(c)
    except:
        pass
print(l)


host = socket.gethostname()
port = 8000

c = CustomSocket(host, port)
c.clientConnect()

cap = cv2.VideoCapture(int(input("Cam index: ")))
cap.set(4, 480)
cap.set(3, 640)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # cv2.imshow('client_cam', frame)

    # print("Send")
    msg = c.req(frame)
    print(msg)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
