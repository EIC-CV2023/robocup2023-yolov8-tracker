from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml
from random import randint
from ultralytics.SORT import *


model = YOLO("yolov8s-pose.pt")

l=[]
for c in range(10):
    try:
        ret, frame = cv2.VideoCapture(c).read()

        if ret: 
            l.append(c)
    except:
        pass
print(l)

start = time.time()
cap = cv2.VideoCapture(int(input("Cam index: ")))

rand_color_list = np.random.rand(20, 3) * 255

while cap.isOpened():
    res = []
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # print("fps: " + str(round(1 / (time.time() - start), 2)))
    start = time.time()
    # frame2 = np.copy(frame)

    results = model.predict(source=frame, conf=0.7, show=True)[0]
  
    print(res)

    # cv2.imshow("frame", frame2)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
