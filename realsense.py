import cv2
import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return True, self.depth_image, self.color_image

    def release(self):
        self.pipeline.stop()

    def get_point_cloud(self):
        self.get_frame()
        height = self.depth_image.shape[0]
        width = self.depth_image.shape[1]

        x_grid, y_grid = np.meshgrid(range(width),range(height))

        point_cloud = np.stack((x_grid, y_grid, self.depth_image), axis=2)

        return True, self.depth_image, self.color_image, point_cloud
         


def main():        
    # Main --
    # Set initial point
    point = (250, 250)

    def show_distance(event, x, y, args, params):
        global point
        point = (x, y)

    # Initialize Camera Intel Realsense
    dc = DepthCamera()

    # Create mouse event
    cv2.namedWindow("Color frame")
    cv2.setMouseCallback("Color frame", show_distance)

    while True:
        ret, depth_frame, color_frame, point_cloud = dc.get_point_cloud()

        # Show distance for a specific point
        cv2.circle(color_frame, point, 4, (0, 0, 255))
        distance = (point[0],point[1],depth_frame[point[1], point[0]])
        cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.imshow("Color frame", color_frame)

        print(point_cloud)

        if cv2.waitKey(10) == ord("q"):
                break
        

if __name__ == '__main__':
    main()