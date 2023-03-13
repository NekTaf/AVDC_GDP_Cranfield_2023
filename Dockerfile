FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

WORKDIR /Desktop/AVDC_GDP_Cranfield_2023

# Get YOLOv8 
ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/

RUN apt-get update \
    && apt-get install --no-install-recommends -y  \
        libgl1-mesa-glx libglib2.0-0 python3 python3-pip \
    && pip3 install ultralytics \
    && rm -rf /var/lib/apt/lists/*



RUN apt-get update && \
    apt-get install -y libv4l-dev


RUN apt-get install libxcb-xinerama0


RUN pip uninstall -y torch torchvision
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


COPY . .

<<<<<<< HEAD
# Run YOLOv8 Python file  
CMD ["python3", "AI/testcam.py"]
=======
ENTRYPOINT ["python3", "AI/testcam.py"]
CMD ["--camera", "/dev/video0"]

>>>>>>> parent of 704e720 (Update Dockerfile)
