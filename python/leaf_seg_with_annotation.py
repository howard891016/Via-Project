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

def overlay_mask_on_image(mask, original_image, color=[0, 255, 0], alpha=0.3):
    print(color)
    green_mask = np.zeros((128, 128, 3), dtype=np.uint8)
    green_mask[:, :, 0] = color[0]
    green_mask[:, :, 1] = color[1]
    green_mask[:, :, 2] = color[2]

    original_image = np.array(original_image)
    other_mask = np.broadcast_to(mask == 1, original_image.shape)
    green_mask[other_mask] = 255
    
    print("putting the layers on")
    try:
        # Attempt to blend the images using cv2.addWeighted
        blended_image = cv2.addWeighted(original_image, 1 - alpha, green_mask, alpha, 0)
    except Exception as e:
        # Print out the error if something goes wrong
        print(f"An error occurred: {e}")
        print("Was able to put the layers on")
    return blended_image
    

def main(mdla_path_bound, mdla_path_segment, mdla_path_detect, image_path, save_path):

    start_time = time.time()

    '''Segmentation'''

    segment = Segment(mdla_path=mdla_path_segment)

    # Initialize model
    ret = segment.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    
    image = Image.open(image_path)
    image = image.resize((128, 128))
    original_image = image.copy()
    
    # Check if the picture has 4 channels
    if image.mode == 'RGBA':
        # 转换为 RGB，去除alpha通道
        image = image.convert('RGB')

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
    
    #color = (255, 0, 0

    
    '''Draw Leaf Bounding Box'''

    bound = Bound(mdla_path=mdla_path_bound)

    # Initialize model
    ret = bound.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    
    # print(segment_img.shape)
    # Load input image
    # image_pil = Image.fromarray(cv2.cvtColor(segment_img, cv2.COLOR_BGR2RGB))

    # 使用 PIL 调整图像大小
    # image = image_pil.resize((128, 128), Image.LANCZOS)

    dtype = np.float32
    segment_img = np.array(segment_img, dtype=dtype)
    input_array = np.expand_dims(segment_img, axis=0)
    # Set input buffer for inference
    bound.SetInputBuffer(input_array, 0)
    # Execute model
    ret = bound.Execute()
    if ret != True:
        print("Failed to Execute")
        return
    
    segment_img = (segment_img * 255).astype(np.uint8)
    # segment_img = (segment_img * 255)
    image_pil = Image.fromarray(segment_img)
    segment_img = image_pil.resize((128, 128), Image.LANCZOS)
    print("before bound")
    bound_img = bound.postprocess(segment_img)

    # print(type(bound_img))
    bound_img = cv2.resize(bound_img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    bound_img_resized = cv2.cvtColor(bound_img, cv2.COLOR_BGR2RGB)
    


 


    '''Detect Leaf Disease'''
    detect = Detect(mdla_path=mdla_path_detect)

    # Initialize model
    ret = detect.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    
    print(bound_img_resized.shape)
    
    # Preprocess input image
    input_array = detect.img_preprocess(bound_img_resized)
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
    print(detect.GetOutputBuffer(0))
    print(class_names[np.argmax(detect.GetOutputBuffer(0))])
    # detect.postprocess(image)

    disease = class_names[np.argmax(detect.GetOutputBuffer(0))]
    end_time = time.time()
    total_time = end_time - start_time
    print(str(total_time) + 's')
    
    color = (255, 0, 0)
    if(disease == "healthy"):
        color = (0,255, 0)
    print("Before adding fancy mask")

    print("Finished segmenting")

    segment_img_with_overlay = overlay_mask_on_image(need, original_image, color)
    annotated_image = bound.draw_bbox_on_image(segment_img_with_overlay, color)
    annotated_image = bound.add_disease_label(annotated_image, disease, 0.9, color)
    

    
    print("Finish annotation")
    # print(type(bound_img))

    # print(type(image_pil))
    # bound_img = image_pil.resize((128, 128), Image.LANCZOS)

    # 使用 PIL 调整图像大小
    
    #save the annotated image to the provided path 
    
    

    # bound_img = cv2.cvtColor(bound_img, cv2.COLOR_RGB2BGR)
    # cv2.imshow("result", annotated_image)
    # cv2.waitKey(1000)
    # cv2.imwrite("annotation.jpg", annotated_image)
    #sv_path = os.path.dirname(image_path)
    sv_path = save_path
    print("SV PATH: " + sv_path)
    image = Image.fromarray(annotated_image)
    print("image:" + image_path)


    #sv_path += "/annotation.jpg"
    #print(sv_path)
    image.save(sv_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Detect model with NeuronHelper')
    parser.add_argument('--dla-bound-path', type=str, default='best_float32.mdla',
                        help='Path to the Bound mdla file')
    parser.add_argument('--dla-segment-path', type=str, default='seg_diff_layer.mdla',
                        help='Path to the Segmentation mdla file')
    parser.add_argument('--dla-detect-path', type=str, default='det_seg_trans_aug_v2.mdla',
                        help='Path to the Detection mdla file')
    parser.add_argument('--image-path', type=str, default='bound_test.JPG',
                        help='Path to the input image')
    parser.add_argument('--save-path', type=str, default='bound_test.JPG',
                        help='Path to save the image')
    args = parser.parse_args()

    main(args.dla_bound_path, args.dla_segment_path, args.dla_detect_path, args.image_path, args.save_path)
