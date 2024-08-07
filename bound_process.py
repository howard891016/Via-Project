from transforma_bound import Bound
from transforma_seg import Segment
from transforma_detect import Detect
from NeuronRuntimeHelper import NeuronContext
from PIL import Image
import argparse
import numpy as np
import cv2
import time
import os

def bound_process(mdla_path_bound, image_path):
    image = Image.open(image_path)

    if image.mode == 'RGBA':
        # 转换为 RGB，去除alpha通道
        image = image.convert('RGB')
    
    
    img_w, img_h = image.size
    back = Image.new("RGB", (img_w, img_w), "black")
    offset = ((img_w - img_w) // 2, (img_w - img_h) // 2)
    back.paste(image, offset)
    img = cv2.cvtColor(np.array(back), cv2.COLOR_RGB2BGR)
    
    
    back_img = back.copy()
    back_img = back_img.resize((640, 640))
    # PIL: RGB, cv2: BGR
    # Preprocess input image
    
    bound = Bound(mdla_path=mdla_path_bound)
    input_array = bound.img_preprocess(back_img)
    # Initialize model
    
    ret = bound.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    
    # Set input buffer for inference
    bound.SetInputBuffer(input_array, 0)

    ret = bound.Execute()
    if ret != True:
        print("Failed to Execute")
        return
    
    bound_output = bound.GetOutputBuffer(0)
    bound_imgs = bound.postprocess(back_img)
 
    img_resized = []
    for bound_img in bound_imgs:
        try:
            bound_img_resized = cv2.resize(bound_img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            img_resized.append(bound_img_resized)
        except:
            print("No leaf")    

    return img_resized, bound_output
  
