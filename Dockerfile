FROM ultralytics/ultralytics:latest-arm64

# Working Directory
WORKDIR /Desktop/AVDC_GDP_Cranfield_2023


## Install correct version of PyTorch
#RUN pip uninstall -y torch torchvision
#RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Copy all directory code into the container
COPY . .

# Run YOLOv8 Python file  
CMD ["python3", "AI/testcam.py"]

