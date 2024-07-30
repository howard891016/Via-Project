from transforma_bound import Bound
from transforma_seg import Segment
from transforma_detect import Detect
from NeuronRuntimeHelper import NeuronContext
from PIL import Image
import argparse
import numpy as np
import cv2
import time

def main(mdla_path, image_path):

    start_time = time.time()

    '''Load Image and Draw Leaf Bounding Box'''

    bound = Bound(mdla_path=mdla_path)

    # Initialize model
    ret = bound.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    
    image = Image.open(image_path)
    image = image.resize((128, 128))
    input_array = bound.img_preprocess(image)
 
    # Set input buffer for inference
    bound.SetInputBuffer(input_array, 0)

    # Execute model
    ret = bound.Execute()
    if ret != True:
        print("Failed to Execute")
        return
    
    bound.postprocess(image)

    '''Segmentation'''

    segment = Segment(mdla_path=mdla_path)

    # Initialize model
    ret = segment.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    
    # Load input image
    image = Image.open(image_path)
    image = image.resize((128, 128))
    
    # Preprocess input image
    input_array = segment.img_preprocess(image)

    # Set input buffer for inference
    segment.SetInputBuffer(input_array, 0)

    # Execute model
    ret = segment.Execute()
    if ret != True:
        print("Failed to Execute")
        return
    
    image = segment.GetOutputBuffer(0)
    need = segment.create_mask(image)
    
    white_images = np.zeros_like(input_array[0])
    wants = np.array([np.where(need == 0, input_array[0], white_images)])
    
    segment.postprocess(wants[0])



    '''Detect Leaf Disease'''
    detect = Detect(mdla_path=mdla_path)

    # Initialize model
    ret = detect.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    
    # Load input image
    image = Image.open(image_path)
    image = image.resize((128, 128))

    # Preprocess input image
    input_array = detect.img_preprocess(image)

    # Set input buffer for inference
    detect.SetInputBuffer(input_array, 0)

    # Execute model
    ret = detect.Execute()
    if ret != True:
        print("Failed to Execute")
        return
    
    output_array = []

    # Postprocess output
    class_names = ['blight','citrus' ,'healthy', 'measles', 'mildew', 'mite', 'mold', 'rot', 'rust', 'scab', 'scorch', 'spot', 'virus']
    print(detect.GetOutputBuffer(0))
    print(class_names[np.argmax(detect.GetOutputBuffer(0))])
    detect.postprocess(image)


    end_time = time.time()
    total_time = end_time - start_time
    print(str(total_time) + 's')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Detect model with NeuronHelper')
    parser.add_argument('--dla-path', type=str, default='det_noseg_v2.mdla',
                        help='Path to the Detection mdla file')
    parser.add_argument('--image-path', type=str, default='mask.jpg',
                        help='Path to the input image')
    args = parser.parse_args()

    main(args.dla_path, args.image_path)
