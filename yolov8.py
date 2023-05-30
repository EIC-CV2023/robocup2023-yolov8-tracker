from ultralytics import YOLO
import cv2
import time
import numpy as np

model = YOLO("yolov8s-seg.pt")

start = time.time()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print("fps: " + str(round(1 / (time.time() - start), 2)))
    start = time.time()

    # print(type(frame))
    results = model.predict(source=frame, conf=0.5, show=True)[0]
    if results.boxes:
        for obj in results.boxes:
            # print(obj.xywhn)
            # print(obj.cls)
            # print(obj.conf)
            # print(obj.data)
            pass
        print(results.boxes.data.cpu().numpy())

    if results.masks:
        print(len(results.masks.xy))
        for obj in results.masks.xy:
            # print(obj)
            # print(obj.data)
            pass
        # print(results.masks.data.cpu().numpy())  # shape (n, h, w)

        # det = results.masks.data[0].numpy()

    if results.probs:
        print(results.probs)

    # cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
