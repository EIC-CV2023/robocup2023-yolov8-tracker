import cv2
from ultralytics import YOLO
import time
import yaml
import numpy as np
import sys
import os
import cv2
from custom_socket import CustomSocket
import socket
import json
import numpy as np
import traceback

from yolov8_track import V8Tracker

WEIGHT = "yolov8s-seg.pt"
DATASET_NAME = "coco"
# DATASET_NAME = {0: "coke"}
# DATASET_NAME = {0: "coke", 1: "milk", 2: "waterbottle"
YOLOV8_CONFIG = {"tracker": "botsort.yaml",
                 "conf": 0.7,
                 "iou": 0.3,
                 "show": False,
                 "verbose": False}



def main():
    HOST = socket.gethostname()
    PORT = 12301

    server = CustomSocket(HOST, PORT)
    server.startServer()

    

    while True:
        # Wait for connection from client :}
        conn, addr = server.sock.accept()
        print("Client connected from", addr)

        start = time.time()
        model = V8Tracker(config=YOLOV8_CONFIG, weight=f"weight/{WEIGHT}", dataset_name="coco")

        # Process frame received from client
        while True:
            res = dict()
            try:
                data = server.recvMsg(conn, has_splitter=True)

                frame_height, frame_width = int(data[0]), int(data[1])
                # print(frame_height, frame_width)
                
                img = np.frombuffer(data[-1], dtype=np.uint8).reshape(frame_height, frame_width, 3)
                # cv2.imwrite("save.jpg", img)

                res = model.track(img, socket_result=True)

                # Send back result
                # print(res)
                server.sendMsg(conn, json.dumps(res))

            except Exception as e:
                traceback.print_exc()
                print(e)
                print("Connection Closed")
                del res
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


    
