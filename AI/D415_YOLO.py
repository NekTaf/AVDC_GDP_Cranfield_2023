#######################################

# YOLO with Camera frames

#######################################


from ultralytics import YOLO
import cv2
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image




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
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # Load YOLO Model
    model = YOLO("best2.pt")

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
                        depth_point = depth_frame.get_distance(x, y)
                        print(depth_point, "for target :",N)

            thr_filter = rs.threshold_filter()
            thr_filter.set_option(rs.option.min_distance, 0.1)
            thr_filter.set_option(rs.option.max_distance, depth_point)
            depth_frame = thr_filter.process(depth_frame)
            depth_image = np.asanyarray(depth_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack Images
            images = np.hstack((res_plotted, depth_colormap))

            # Show images
            cv2.namedWindow('YOLO', cv2.WINDOW_AUTOSIZE)

            # Plot Output image
            cv2.imshow('YOLO', images)

            # Check if the user pressed the "q" key to quit
            if cv2.waitKey(1) == ord('q'):
                break



    finally:

        # Stop streaming
        pipeline.stop()

        # Release the camera and destroy all windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
