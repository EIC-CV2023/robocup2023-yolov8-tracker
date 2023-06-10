import cv2
from ultralytics import YOLO
import time
import yaml
import numpy as np

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


class V8Tracker:
    def __init__(self, config, weight="weight/yolov8s-seg.pt", dataset_name="coco"):
        self.config = config
        self.weight = weight
        self.model = YOLO(weight)

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

    def track(self, frame, socket_result=False):
        self.frame_height, self.frame_width = frame.shape[:-1]
        track_results = self.model.track(source=frame,
                                         conf=self.config["conf"],
                                         iou=self.config["iou"],
                                         show=self.config["show"],
                                         persist=True,
                                         verbose=self.config["verbose"],
                                         tracker=self.config["tracker"],
                                         imgsz=frame.shape[:-1])[0]

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

            if not socket_result:

                self.results[tracker_id] = {"name": obj_name,
                                            "box": box.xywh.numpy()[0],
                                            "mask": mask.data.cpu().numpy()[0]}
            else:
                self.results[tracker_id] = {"name": obj_name,
                                            "box": box.xywh.numpy()[0].astype("uint8").tolist(),
                                            "mask": mask.data.cpu().numpy()[0].ravel().astype("uint8").tolist()}

        return self.results

    def get_mask_frame(self):
        mask_frame = np.zeros((self.frame_height, self.frame_width))

        for obj_id in self.results:
            mask = self.results[obj_id]["mask"]

            mask_frame += mask * obj_id

        return mask_frame

    def draw_mask_frame(self):
        return cv2.normalize(src=self.get_mask_frame(), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)


def main():
    cap = cv2.VideoCapture(list_available_cam(10))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    start = time.time()

    model = V8Tracker(config=YOLOV8_CONFIG,
                      weight=f"weight/{WEIGHT}", dataset_name="coco")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error")

        cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        start = time.time()

        yolo_result = model.track(frame, socket_result=True)

        print(yolo_result)
        print(frame.shape)

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
