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

WEIGHT = "rbc2023-dday-seg.pt"
# DATASET = "coco"
DATASET = ['apple', 'banana', 'baseball', 'bowl', 'cheezit', 'chocolate_cornflakes', 'chocolate_jello', 'cleanser', 'coffee_grounds', 'cola', 'cornflakes', 'cup', 'dice', 'fork', 'iced_tea', 'juice_pack', 'knife', 'lemon', 'milk', 'mustard',
           'orange', 'orange_juice', 'peach', 'pear', 'plate', 'plum', 'pringles', 'red_wine', 'rubiks_cube', 'soccer_ball', 'spam', 'sponge', 'spoon', 'strawberry', 'strawberry_jello', 'sugar', 'tennis_ball', 'tomato_soup', 'tropical_juice', 'tuna']
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
        model = V8Tracker(config=YOLOV8_CONFIG,
                          weight=f"weight/{WEIGHT}", dataset_name=DATASET)

        # Process frame received from client
        while True:
            res = dict()
            msg = {"res": res}
            try:
                data = server.recvMsg(
                    conn, has_splitter=True)
                frame_height, frame_width, frame = data

                msg["camera_info"] = [frame_width, frame_height]

                yolo_res = model.track(frame, socket_result=True)

                msg["res"] = yolo_res

                # Send back result
                # print(res)
                server.sendMsg(conn, json.dumps(msg))

            except Exception as e:
                traceback.print_exc()
                print(e)
                print("Connection Closed")
                del res
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
