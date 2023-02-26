#######################################

### Intel Realsense 415 Get Depth

#######################################

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
                 "  3 for only Depth \n")



try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_point = depth_frame.get_distance(320, 240)

        print(depth_point)

        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


        # Get Width and Height
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape


        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)


        if mode_selection =='1': cv2.imshow('RealSense', images)
        elif mode_selection =='2': cv2.imshow('RealSense', color_image)
        elif mode_selection == '3': cv2.imshow('RealSense', depth_colormap)

        #cv2.imshow('RealSense', images)


        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()





