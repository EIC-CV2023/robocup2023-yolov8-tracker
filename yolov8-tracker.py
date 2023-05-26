import argparse
import json
import os
import socket
from pickle import NONE
from custom_socket import CustomSocket
import numpy as np
import time
from ultralytics.SORT import *
import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml


WEIGHT = "yolov8s-seg.pt"
# WEIGHT = "all2ndvidbest.pt"
DATASET_NAME = "coco"
# DATASET_NAME = {0: "coke"}
# DATASET_NAME = {0: "coke", 1: "milk", 2: "waterbottle"}


class V8Tracker:
    def __init__(self, weight="yolov8s-seg.pt", conf=0.7, dataset_name="coco", sort_max_age=10, sort_min_hits=5, sort_iou_thresh=0.2, show_result=False):
        self.tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits,
                            iou_threshold=sort_iou_thresh)
        self.model = YOLO(weight)
        self.rand_color_list = np.random.rand(20, 3) * 255
        self.conf = conf
        self.show_result = show_result

        if dataset_name == "coco":
            with open("ultralytics/yolo/data/datasets/coco8-seg.yaml", "r") as stream:
                try:
                    datasets = yaml.safe_load(stream)
                    self.datasets_names = datasets['names']
                except:
                    print("No file found")
                    self.datasets_names = ""
        else:
            # In format of {0: name0, 1: name1, ...}
            self.datasets_names = dataset_name

    def draw_box(self, img, bbox, id=None, label=None):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      self.rand_color_list[id % 20], 3)
        cv2.putText(img, f"{id}:{label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1, self.rand_color_list[id % 20], 2)
        return img

    def track(self, frame):
        self.res = []
        self.frame = np.copy(frame)
        self.results = self.model.predict(
            source=self.frame, conf=self.conf, show=self.show_result)[0]
        if self.results.boxes:
            # print(f"DETECT {len(results.boxes)}")
            output = dict()
            dets_to_sort = np.empty((0, 6))

            for i, obj in enumerate(self.results.boxes):
                x1, y1, x2, y2, conf, cls = obj.data.cpu().detach().numpy()[0]
                name = self.datasets_names[int(
                    cls)] if self.datasets_names else 'unknown'

                output[i] = [name, x1, y1, x2, y2]

                dets_to_sort = np.vstack((dets_to_sort,
                                          np.array([x1, y1, x2, y2, conf, cls])))
            # print(dets_to_sort)

            tracked_dets = self.tracker.update(dets_to_sort)
            # print(tracked_dets)
            for tk in tracked_dets:
                x1, y1, x2, y2 = (int(p) for p in tk[:4])
                w, h = x2-x1, y2-y1
                id = int(tk[8])
                cls = tk[4]
                name = self.datasets_names[cls]
                self.draw_box(self.frame, (x1, y1, x2, y2), id, name)
                self.res.append([id, cls, name, x1, y1, w, h])

        return self.res, self.frame


def main():
    HOST = "0.0.0.0"
    # HOST = "192.168.8.99"
    PORT = 8000

    server = CustomSocket(HOST, PORT)
    server.startServer()

    V8T = V8Tracker(weight=WEIGHT, dataset_name=DATASET_NAME,
                    show_result=True)

    while True:
        conn, addr = server.sock.accept()
        print("Client connected from", addr)
        while True:
            try:
                data = server.recvMsg(conn)
                # img = np.frombuffer(data, dtype=np.uint8).reshape(360, 640, 3)
                # img = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)
                img = np.frombuffer(data, dtype=np.uint8).reshape(720, 1280, 3)

                sol, drawn_frame = V8T.track(img)

                out = {}
                obj = []
                formatted_bbox = []

                for s in sol:
                    id, cls, classname, x, y, w, h = s
                    obj.append([id, cls, classname, x, y, w, h])
                    formatted_bbox.append([classname, (x, y, w, h), False])
                out["result"] = obj
                out["n"] = len(obj)
                print(out)
                server.sendMsg(conn, json.dumps(out, indent=4))

                cv2.imshow("Result image", drawn_frame)
                cv2.waitKey(1)

            except Exception as e:
                print(e)
                print("CONNECTION CLOSED")
                break
        # server.stopServer()


if __name__ == '__main__':
    main()
