from transforma_bound import Bound
from transforma_seg import Segment
from transforma_detect import Detect
from NeuronRuntimeHelper import NeuronContext
from PIL import Image
import argparse
import numpy as np
import cv2
import time

def main(mdla_path_bound, mdla_path_segment, mdla_path_detect, image_path):

    start_time = time.time()

    '''Load Image and Draw Leaf Bounding Box'''

    bound = Bound(mdla_path=mdla_path_bound)

    # Initialize model
    ret = bound.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    
    image = Image.open(image_path)
    image = image.resize((128, 128))
    
    # Check if the picture has 4 channels
    if image.mode == 'RGBA':
        # 转换为 RGB，去除alpha通道
        image = image.convert('RGB')

    input_array = bound.img_preprocess(image)

 
    # Set input buffer for inference
    bound.SetInputBuffer(input_array, 0)
    # Execute model
    ret = bound.Execute()
    if ret != True:
        print("Failed to Execute")
        return
    
    bound_img = bound.postprocess(image)
    # bound_img = cv2.cvtColor(bound_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", bound_img)
    cv2.waitKey(1000)

    '''Segmentation'''

    segment = Segment(mdla_path=mdla_path_segment)

    # Initialize model
    ret = segment.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    
    # Load input image
    image_pil = Image.fromarray(cv2.cvtColor(bound_img, cv2.COLOR_BGR2RGB))

    # 使用 PIL 调整图像大小
    image = image_pil.resize((128, 128), Image.LANCZOS)

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
    
    segment_img = segment.postprocess(wants[0])



    '''Detect Leaf Disease'''
    detect = Detect(mdla_path=mdla_path_detect)

    # Initialize model
    ret = detect.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    
    print(segment_img.shape)

    # Preprocess input image
    input_array = detect.img_preprocess(segment_img)
    print(input_array.shape)
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
    # print(detect.GetOutputBuffer(0))
    print(class_names[np.argmax(detect.GetOutputBuffer(0))])
    # detect.postprocess(image)


    end_time = time.time()
    total_time = end_time - start_time
    print(str(total_time) + 's')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Detect model with NeuronHelper')
    parser.add_argument('--dla-bound-path', type=str, default='best_float32.mdla',
                        help='Path to the Bound mdla file')
    parser.add_argument('--dla-segment-path', type=str, default='seg_diff_layer.mdla',
                        help='Path to the Segmentation mdla file')
    parser.add_argument('--dla-detect-path', type=str, default='det_noseg_v2.mdla',
                        help='Path to the Detection mdla file')
    parser.add_argument('--image-path', type=str, default='bound_test.JPG',
                        help='Path to the input image')
    args = parser.parse_args()

    main(args.dla_bound_path, args.dla_segment_path, args.dla_detect_path, args.image_path)
