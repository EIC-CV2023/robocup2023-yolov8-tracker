from ultralytics import YOLO
import cv2
import time
import numpy as np


rand_color = np.random.rand(40, 3) * 255

def draw_mask(frame, mask, id, opa=0.4):
    color = rand_color[id]
    frame[mask] = frame[mask] * (1-opa) + color * (opa)


img = cv2.resize(cv2.imread("group-photo.jpg"), (1440,800))
print(img.shape)

model = YOLO("yolov8s-seg.pt")
results = model.predict(source=img, conf=0.3, show=True, imgsz=1440)[0]

for i, obj_mask in enumerate(results.masks.data.cpu().numpy()):
    print(obj_mask.shape)
    print(obj_mask)
    draw_mask(img, obj_mask.astype(bool), i, 0.6)

    
cv2.imshow("i", img)
cv2.waitKey()
