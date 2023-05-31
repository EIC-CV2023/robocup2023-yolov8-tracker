import cv2
from ultralytics import YOLO
import time

def list_available_cam(max_n):
    list_cam = []
    for n in range(max_n):
        cap = cv2.VideoCapture(n)
        ret, _ = cap.read()

        if ret:
            list_cam.append(n)
        cap.release()
    return list_cam

print(list_available_cam(10))

cap = cv2.VideoCapture(int(input("Cam index: ")))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

model = YOLO("yolov8s-seg.pt")
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error")

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    start = time.time()


    results = model.track(source=frame, conf = 0.7, iou = 0.5, show=True, persist=True, tracker="botsort.yaml")

    print(results)

    


