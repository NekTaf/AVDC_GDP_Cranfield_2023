FROM ultralytics/ultralytics:latest-arm64

WORKDIR /Desktop/AVDC_GDP_Cranfield_2023



RUN apt-get update && \
    apt-get install -y libv4l-dev


#RUN apt-get update && \
  #  apt-get install -y libqt5gui5 && \
 #   rm -rf /var/lib/apt/lists/*
#ENV QT_DEBUG_PLUGINS=1


#RUN pip uninstall -y torchvision uninstall torchvision && \
#     pip install torchvision==0.2.0


RUN apt-get install libxcb-xinerama0



#RUN apt-get remove -y libqt5xcbqpa5
#RUN apt-get install -y libqt5xcbqpa5

#RUN pip uninstall -y opencv-python 
#RUN pip install opencv-python-headless

#RUN apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev


RUN pip uninstall -y torch torchvision
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116





#QT_DEBUG_PLUGINS=1

COPY . .

ENTRYPOINT ["python3", "AI/testcam.py"]
CMD ["--camera", "/dev/video0"]

