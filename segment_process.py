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

def segment_process(mdla_path_segment):
    segment = Segment(mdla_path=mdla_path_segment)

    # Initialize model
    ret = segment.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return

    
    
    # Check if the picture has 4 channels
    # bound_img_resized = Image.fromarray(cv2.cvtColor(bound_img_resized, cv2.COLOR_BGR2RGB))
    
    input_array = segment.img_preprocess(bound_img_resized)
    
    
    # Set input buffer for inference
    segment.SetInputBuffer(input_array, 0)
    seg_model_start = time.time()
    print("Start Seg model: " + str(seg_model_start - start_time) + "ms")
    
    # Execute model
    ret = segment.Execute()
    if ret != True:
        print("Failed to Execute")
    
    seg_model_end = time.time()
    print("End Seg model: " + str(seg_model_end - start_time) + "ms")
    
    image = segment.GetOutputBuffer(0)
    need = segment.create_mask(image)
    
    white_images = np.zeros_like(input_array[0])
    wants = np.array([np.where(need == 0, input_array[0], white_images)])
        
    # print(wants[0].mode)
    # segment_img = segment.postprocess(wants[0])
    
    # segment_img = wants[0]
    segment_img = bound_img_resized


    