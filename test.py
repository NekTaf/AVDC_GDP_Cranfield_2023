from ultralytics import YOLO
import os
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda




def main():
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(0.35)

    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    #model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    path = r"C:\Users\Nekta\PycharmProjects\GDP_AI\data_car"
    os.chdir(path)

    print("Current working directory: {0}".format(os.getcwd()))

    # Use the model
    model.train(data=r"C:\Users\Nekta\PycharmProjects\GDP_AI\data_car\data.yaml", epochs=40, batch=4)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set

    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    success = model.export(format="onnx")  # export the model to ONNX format



def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


if __name__ == '__main__':
    free_gpu_cache()
    main()








