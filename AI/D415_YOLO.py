#!/usr/bin/env python3

"""
intel Realsense D415 using YOLOv8 (pytorch)

Features:

    Car detection using RGB frames

    Semantic segmentation using Depth map of enviroment
"""


from ultralytics import YOLO
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from cluster import Clustering


def clamp(n, min_value, max_value):
    return max(min(n, max_value), min_value)


def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    # Enable RGB and Depth Streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Load YOLO Model
    model = YOLO("best2.pt")

    no_targets_selected=None

    # Start Pipeline
    pipeline.start(config)

    try:
        # Start a loop to continuously capture frames and display them
        while True:

            pipeline.poll_for_frames()

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Format Depth Frame (for get distance)
            depth_frame = depth_frame.as_depth_frame()

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # # Get Width and Height
            # depth_colormap_dim = depth_colormap.shape
            # color_colormap_dim = color_image.shape

            # Convert to RBG to BGR (PIL is BGR format)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Convert frame to PIL, or else lag in YOLO when using D415???
            img = Image.fromarray(color_image)

            # YOLOv8n custom on Frame
            results = model(img)

            # Plot Results
            res_plotted = results[0].plot(show_conf=True)

            # Array Nx1 for confidence of object detected
            conf_array = results[0].boxes.conf.cpu().numpy()

            no_targets_total=len(conf_array)

            # Array storing ranges for each target
            depth_point = np.empty(no_targets_total)


            # Array Nx4 for xywh of bounding box for object detected
            box_array = results[0].boxes.xywh.cpu().numpy()

            # Check if Object is detected and get centre
            if box_array.size > 0:

                # For each detected Target
                for N in range(len(conf_array)):

                    # Plot dot only if above confidence percentage
                    if conf_array[N] > 0.1:
                        x = int(box_array[N, 0])
                        y = int(box_array[N, 1])

                        # Draw point at centre of detected objects
                        radius = 5
                        color = (0, 255, 0)  # BGR format
                        res_plotted = cv2.circle(res_plotted, (x, y), radius, color, -1)

                        # Get Depth for Center Coordinate
                        depth_point[N] = depth_frame.get_distance(x, y)

                        # Clamp Target range to D415 allowed ranges
                        if depth_point[N] > 16: print(depth_point, " Target range suppresses 16m")

                        depth_point[N]=clamp(depth_point[N], 0, 16)

                        print(depth_point[N], " Range for target :", N)

                        print(conf_array[N], " Confidence for target :", N)

                no_targets_selected=len(depth_point)

                print(no_targets_selected)

            # Check if array empty (Targets Detected)
            if depth_point.size != 0:
                # Threshold Filter for Segmentation
                thr_filter = rs.threshold_filter()
                thr_filter.set_option(rs.option.min_distance, 0.1)
                thr_filter.set_option(rs.option.max_distance, max(depth_point))
                depth_frame = thr_filter.process(depth_frame)
                depth_frame = rs.hole_filling_filter().process(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


            depth_colormap_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
            # print(depth_colormap_gray.shape)
            # print(np.unique(depth_colormap_gray))

            # Show images
            cv2.namedWindow('YOLO', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('YOLO', res_plotted)

            cv2.namedWindow('Gray Depth Map', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Gray Depth Map', cv2.equalizeHist(depth_colormap_gray))

            if not no_targets_selected:
                print("No targets with high probability detected")

            else:
                out =Clustering(depth_colormap_gray,no_targets_selected+1).KM()

                cv2.namedWindow('Gray K-Means', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Gray K-Means', out)

            # Check if the user pressed the "q" key to quit
            if cv2.waitKey(1) == ord('q'):
                break



    finally:

        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
