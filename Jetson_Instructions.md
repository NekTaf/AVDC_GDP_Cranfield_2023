JETSON Nano Docker 

Jetson YOLOv8 Docker

Make Docker 

------  sudo docker build -t yolov8n .



Run Docker

------  sudo docker run --privileged --device=/dev/video*:/dev/video*  yolov8n



View all Dockers

------  sudo docker ps -a



View all Dockers 

------  docker ps -a





pyrealsense2 instructions (PyPi pip not compatible with ARM on Jetson nano)

Build from source:
        https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#building-from-source

Set pyrealsense2 path: (execute in appropriate directory) 

------  export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.x/pyrealsense2 (replace with python version ex. python3.6)
