#!/usr/bin/env python3

"""
intel Realsense D415 depth map and RGB frames example script
"""


import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from tkinter import simpledialog

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

# Start streaming
pipeline.start(config)

# GUI change between RBG  &  Depth
ROOT = tk.Tk()
ROOT.withdraw()

# the input dialog
mode_selection = simpledialog.askstring(title="Mode",
                                        prompt="Choose: \n"
                                               "  1 for RGB and Depth\n"
                                               "  2 for only RGB \n"
                                               "  3 for only Depth \n"
                                        "  4 for only HSV \n")

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue


        # Filters
        depth_frame = rs.decimation_filter(1).process(depth_frame)
        depth_frame = rs.disparity_transform(True).process(depth_frame)
        depth_frame = rs.spatial_filter().process(depth_frame)
        depth_frame = rs.temporal_filter().process(depth_frame)
        depth_frame = rs.disparity_transform(False).process(depth_frame)
        # thr_filter = rs.threshold_filter()
        # thr_filter.set_option(rs.option.min_distance, 0)
        # thr_filter.set_option(rs.option.max_distance, 4)
        # depth_frame = thr_filter.process(depth_frame)
        # depth_frame = rs.hole_filling_filter().process(depth_frame)

        # Format Depth Frame (for get distance)
        depth_frame = depth_frame.as_depth_frame()

        # Get Depth for Center Coordinate
        depth_point = depth_frame.get_distance(320, 240)
        print(depth_point)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Get Width and Height
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # Stack Images
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        if mode_selection == '1':
            cv2.imshow('RealSense', images)
        elif mode_selection == '2':
            cv2.imshow('RealSense', color_image)
        elif mode_selection == '3':
            cv2.imshow('RealSense', gray)
        elif mode_selection == '4':
            cv2.imshow('RealSense', hsv)


        # cv2.imshow('RealSense', images)

        # print(depth_colormap)

        # print((gray.shape))

        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
