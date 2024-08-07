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

def detect_process(mdla_path_detect, img_resized):
    
    detect = Detect(mdla_path=mdla_path_detect)

    # Initialize model
    ret = detect.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return

    # print(bound_img_resized.shape)
    # Preprocess input image
    
    output_array = []
    class_names = ['blight','citrus' ,'healthy', 'measles', 'mildew', 'mite', 'mold', 'rot', 'rust', 'scab', 'scorch', 'spot', 'virus']
  
    
    for bound_img in img_resized:
        input_array = detect.img_preprocess(bound_img)
        # Set input buffer for inference
        detect.SetInputBuffer(input_array, 0)
    
        # Execute model
        ret = detect.Execute()
        if ret != True:
            print("Failed to Execute")
            return

        
        
        # print(detect.GetOutputBuffer(0))
        print(class_names[np.argmax(detect.GetOutputBuffer(0))])
        output_array.append(class_names[np.argmax(detect.GetOutputBuffer(0))])

        disease = class_names[np.argmax(detect.GetOutputBuffer(0))]
        
    return output_array 