#######################################

# YOLO with Camera frames

#######################################


from ultralytics import YOLO
import cv2
import time
import cv2
import numpy as np

model = YOLO("best2.pt")

# Create a VideoCapture object to capture images from the camera
cap = cv2.VideoCapture(2)  # 0 is the index of the default camera

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Start a loop to continuously capture frames and display them
while True:

    # Capture a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Failed to capture frame")
        break

    # YOLOv8n custom on Frame
    results = model(frame)

    # List of Objects
    # for r in results:
    #     boxes = r.boxes  # Boxes object for bbox outputs
    #     masks = r.masks  # Masks object for segmenation masks outputs
    #     # probs = r.probs  # Class probabilities for classification outputs

    # Plot Results
    res_plotted = results[0].plot(show_conf=True)

    # Array Nx1 for confidence of object detected
    conf_array = results[0].boxes.conf.cpu().numpy()
    print(conf_array.shape)

    # Array Nx4 for xywh of bounding box for object detected
    box_array = results[0].boxes.xywh.cpu().numpy()

    # Check if Object is detected and get centre
    if box_array.size > 0:

        # For each detected Target
        for N in range(len(conf_array)):

            # Plot dot only if above confidence percentage
            if conf_array[N] > 0.7:
                x = int(box_array[N, 0])
                y = int(box_array[N, 1])

                # Draw point at centre of detected objects
                radius = 5
                color = (0, 255, 0)  # BGR format
                res_plotted = cv2.circle(res_plotted, (x, y), radius, color, -1)

    # Plot Output image
    cv2.imshow('YOLO', res_plotted)

    # Check if the user pressed the "q" key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
