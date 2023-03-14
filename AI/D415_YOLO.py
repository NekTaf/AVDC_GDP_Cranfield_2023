#!/usr/bin/env python3

"""
intel Realsense D415 using YOLOv8 (pytorch)

Features:

    Car detection using RGB frames

    Semantic segmentation using Depth map of environment
"""

from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
from cluster import Clustering
from PIL import Image
from track import noiseFilt
from image_filters import Filtering
import matplotlib.pyplot as pp
import time


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

    # Enable RGB and Depth Streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    # Load YOLO Model
    model = YOLO("best2.pt")

    # Initialise empty
    depth_point = np.empty(0)

    # Start Pipeline
    pipeline.start(config)

    try:
        # Start a loop to continuously capture frames and display them
        while True:

            # Get Latest Frame
            pipeline.poll_for_frames()

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Filters
            color_image = np.asanyarray(color_frame.get_data())
            color_image = color_image[:, 50:640]

            depth_frame = rs.decimation_filter(1).process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            depth_frame = rs.hole_filling_filter().process(depth_frame)

            # Get Depth Information
            depth_frame = depth_frame.as_depth_frame()

            # Get Depth Image
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = depth_image[:, 50:640]

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first (alpha=0.03))
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_BONE)

            # Grayscale Depth Color Map
            depth_colormap_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
            depth_colormap_gray = cv2.equalizeHist(depth_colormap_gray)

            # Bilateral filter to reduce noise
            depth_colormap_gray = cv2.bilateralFilter(depth_colormap_gray, 11, 60, 60)

            input = color_image

            ############################################################################################################
            # # Edge Detectors and dilation
            # edges = cv2.Canny(depth_colormap_gray, 100, 85)
            #
            # # Dilate Edge Lines (Connect unconnected lines
            # kernel = np.ones((10, 10), np.uint8)
            # edges = cv2.dilate(edges, kernel, iterations=2)
            #
            # depth_colormap_gray = cv2.bitwise_and(depth_colormap_gray, depth_colormap_gray, mask=edges)

            # binary_mask = np.where(depth_colormap_gray > 0, 1, 0)
            # binary_mask = np.uint8(binary_mask)

            # # Apply mask
            # masked_img = cv2.bitwise_and(color_image, color_image, mask=binary_mask)
            ############################################################################################################

            ############################################################################################################
            # # Cluster Depth Map
            # clustered_depth_map = Clustering(depth_colormap_gray, 2).KM()
            #
            # # Cluster Values
            # cluster_vals = np.unique(clustered_depth_map)
            #
            # # Create Binary Mask from Previous Segmentations
            # binary_mask = np.where(clustered_depth_map < (cluster_vals[0] + 5), 1, 0)
            # binary_mask = np.uint8(binary_mask)
            #
            # # Apply mask
            # input = cv2.bitwise_and(color_image, color_image, mask=binary_mask)
            ############################################################################################################

            # YOLOv8n custom on Frame
            results = model(Image.fromarray(input))

            # Plot Results
            res_plotted = results[0].plot(show_conf=True)

            res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

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

                        print(" Range for target", N, ":", depth_point[N], )

                        print(" Confidence for target", N, ":", conf_array[N])

            # Show images
            cv2.namedWindow('RGB Image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB Image', (color_image))

            cv2.namedWindow('YOLO', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('YOLO', res_plotted)

            cv2.namedWindow('Depth Map', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Depth Map', depth_colormap)
            #
            cv2.namedWindow('Depth Map Mask', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Depth Map Mask', cv2.equalizeHist(depth_colormap_gray))

            # Check if the user pressed the "q" key to quit
            if cv2.waitKey(1) == ord('q'):
                break


    finally:

        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
