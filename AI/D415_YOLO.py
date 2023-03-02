#!/usr/bin/env python3

"""
intel Realsense D415 using YOLOv8 (pytorch)

Features:

    Car detection using RGB frames

    Semantic segmentation using Depth map of environment
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

    # Initialise empty
    depth_point = np.empty(0)

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

            # Check if array empty (Targets Detected)
            if depth_point.size != 0 and max(depth_point)>0.15:
                # Threshold Filter for Segmentation
                threshold_filter = rs.threshold_filter(min_dist=0.15, max_dist= max(depth_point))
                depth_frame = threshold_filter.process(depth_frame).as_depth_frame()


            # Get Depth Image
            depth_image = np.asanyarray(depth_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Grayscale Depth Color Map
            depth_colormap_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)

            # Cluster Depth Map
            clustered_depth_map = Clustering(depth_colormap_gray, 2).KM()

            # print(np.unique(np.uint8(clustered_depth_map)))

            # Cluster Values
            cluster_vals = np.unique(clustered_depth_map)

            # Conve
            binary_mask = np.where(clustered_depth_map < (cluster_vals[0] + 5), 1, 0)
            binary_mask = np.uint8(binary_mask)

            # Apply Morphological filter to cover holes
            # kernel = np.ones((9, 9), np.uint8)
            # mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

            # Define the structuring element
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph_filtered = cv2.dilate(binary_mask, kernel, iterations=1)


            masked_img = cv2.bitwise_and(color_image, color_image, mask=morph_filtered)


            # masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

            # YOLOv8n custom on Frame
            results = model(masked_img)

            # Plot Results
            res_plotted = results[0].plot(show_conf=True)

            # Array Nx1 for confidence of object detected
            conf_array = results[0].boxes.conf.cpu().numpy()

            no_targets_total = len(conf_array)

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

                        depth_point[N] = clamp(depth_point[N], 0, 16)

                        print(depth_point[N], " Range for target :", N)

                        print(conf_array[N], " Confidence for target :", N)

                no_targets_selected = len(depth_point)

                print("Total number of selected targets",no_targets_selected)

            # Show images
            cv2.namedWindow('RGB Image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB Image', color_image)

            cv2.namedWindow('YOLO', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('YOLO', res_plotted)

            cv2.namedWindow('MASK', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('MASK', cv2.equalizeHist(morph_filtered))

            cv2.namedWindow('Gray Depth Map', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Gray Depth Map', cv2.equalizeHist(depth_colormap_gray))

            # Check if the user pressed the "q" key to quit
            if cv2.waitKey(1) == ord('q'):
                break



    finally:

        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
