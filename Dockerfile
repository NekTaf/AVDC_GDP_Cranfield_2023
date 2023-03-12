FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Working Directory
WORKDIR /Desktop/AVDC_GDP_Cranfield_2023

# Get YOLOv8 
ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/

RUN apt-get update \
    && apt-get install --no-install-recommends -y  \
        libgl1-mesa-glx libglib2.0-0 python3 python3-pip \
    && pip3 install ultralytics \
    && rm -rf /var/lib/apt/lists/*



# Install correct version of PyTorch
RUN pip uninstall -y torch torchvision
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Copy all directory code into the container
COPY . .

# Run YOLOv8 Python file  
CMD ["python3", "AI/testcam.py"]
