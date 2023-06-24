import socket
import cv2
import numpy as np
import time
from custom_socket import CustomSocket
import json


def list_available_cam(max_n):
    list_cam = []
    for n in range(max_n):
        cap = cv2.VideoCapture(n)
        ret, _ = cap.read()

        if ret:
            list_cam.append(n)
        cap.release()
    
    if len(list_cam) == 1:
        return list_cam[0]
    else:
        print(list_cam)
        return int(input("Cam index: "))

# def get_mask_frame(results, dim):
#     mask_frame = np.zeros((dim[0], dim[1]))

#     for obj_id in results:
#         mask = np.reshape(results[obj_id]["mask"], dim).astype("int8")

#         mask_frame += mask * int(obj_id)

#     return mask_frame
    
    
# def draw_mask_frame(mask_frame):
#     return cv2.normalize(src=mask_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

def seg_frame(frame, results):
    frame_dim = frame.shape[:-1]
    blank = np.zeros((*frame_dim, 3), dtype="uint8")

    for obj_id in results:
        obj = results[obj_id]
        mask = np.reshape(obj["mask"], frame_dim).astype(bool)
        blank[mask] = frame[mask]

    return blank



host = socket.gethostname()
port = 12301

c = CustomSocket(host, port)
c.clientConnect()

cap = cv2.VideoCapture(list_available_cam(10))

DIM = (640, 480)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.resize(frame, DIM)
    
    # send image through request and 
    msg = c.req(frame)

    print(len(msg))
    

    # print out all the object detected in the frame.
    for i, obj_id in enumerate(msg):
        obj = msg[obj_id]
        print(f"[{i}][{obj['name']}]")
        print(f"[{i}][{obj['box']}]")
        print(f"[{i}][{len(obj['mask'])}]")

    seg_f = seg_frame(frame, msg)
        
    cv2.imshow("blank",seg_f)
    cv2.imshow("frame", frame)

    # print(msg)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
