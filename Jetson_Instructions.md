JETSON Nano Docker and pyrealsense2 instructions


Jetson YOLOv8 Docker

Make Docker 
      sudo docker build -t yolov8n .

Run Docker
       sudo docker run --privileged --device=/dev/video1:/dev/video1  yolov8n


Jetson set pyrealsense2 path: (execute in appropriate directory) 

export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2
