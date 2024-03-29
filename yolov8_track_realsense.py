import cv2
from ultralytics import YOLO
import time
import yaml
import numpy as np
import pyrealsense2 as rs
from realsense import DepthCamera
import json

from yolov8_track import V8Tracker

WEIGHT = "yolov8s-seg.pt"
DATASET_NAME = "coco"
# DATASET_NAME = {0: "coke"}
# DATASET_NAME = {0: "coke", 1: "milk", 2: "waterbottle"
YOLOV8_CONFIG = {"tracker": "botsort.yaml",
                 "conf": 0.7,
                 "iou": 0.3,
                 "show": True,
                 "verbose": False}



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


class V8Tracker_Realsense(V8Tracker):
    def __init__(self,config,weight="weight/yolov8s-seg.pt", dataset_name="coco"):
        super().__init__(config, weight, dataset_name)

    def track_get_pointcloud(self, frame, pointcloud):
        self.frame_height, self.frame_width = frame.shape[:-1]
        track_results = self.model.track(source=frame, 
                              conf=self.config["conf"], 
                              iou=self.config["iou"], 
                              show=self.config["show"], 
                              persist=True, 
                              verbose=self.config["verbose"], 
                              tracker=self.config["tracker"])[0]
        
        self.results = dict()

        if not track_results:
            # print(track_results)
            return self.results

        for index, result in enumerate(zip(track_results.boxes, track_results.masks)):
            box, mask = result

            if not box.id:
                continue

            tracker_id = int(box.id.numpy()[0])
            
            obj_class = int(box.cls.numpy()[0])
            obj_name = self.datasets_names[obj_class] if self.datasets_names else 'unknown'

            seg_mask = mask.data.cpu().numpy()[0].astype(bool)
            
            # print("count true:", np.count_nonzero(seg_mask))

            seg_pointcloud = pointcloud[seg_mask]
            # print(seg_pointcloud.shape)

            self.results[tracker_id] = {"name": obj_name,
                                        "box": box.xywh.numpy()[0],
                                        "mask": seg_mask,
                                        "pointcloud": seg_pointcloud}

        return self.results


def main():
    start = time.time()

    model = V8Tracker_Realsense(config=YOLOV8_CONFIG, weight=f"weight/{WEIGHT}", dataset_name="coco")
    dc = DepthCamera()

    while True:
        ret, depth_frame, color_frame, point_cloud = dc.get_point_cloud()

        if not ret:
            print("Error")

        yolo_result = model.track_get_pointcloud(color_frame, point_cloud)

        print(yolo_result)

        cv2.putText(color_frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(color_frame.shape[0]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        start = time.time() 

        cv2.imshow("frame", color_frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()