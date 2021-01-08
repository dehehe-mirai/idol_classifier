import os
import cv2
import json
from core.detector import LFFDDetector

"""
    !!! IMPORTANT !!!
    Disable auto-tuning
    You might experience a major slow-down if you run the model on images with varied resolution / aspect ratio.
    This is because MXNet is attempting to find the best convolution algorithm for each input size, 
    so we should disable this behavior if it is not desirable.
""" 
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

IMAGE_PATH = "raw-fhd/Ayumu/4005374.png"
CONFIG_PATH = "configs/anime.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
detector = LFFDDetector(config, use_gpu=False)
image = cv2.imread(IMAGE_PATH)
boxes = detector.detect(image)
print(boxes)