FROM ultralytics/ultralytics:latest-arm64

WORKDIR /Desktop/AVDC_GDP_Cranfield_2023


RUN apt-get update && \
    apt-get install -y libv4l-dev


RUN apt-get install libxcb-xinerama0


RUN pip uninstall -y torch torchvision
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


COPY . .

ENTRYPOINT ["python3", "AI/testcam.py"]
CMD ["--camera", "/dev/video0"]

