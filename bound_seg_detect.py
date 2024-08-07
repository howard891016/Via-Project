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
    # print(color)
    green_mask = np.zeros((128 * 3, 128 * 3, 3), dtype=np.uint8)
    green_mask[:, :, 0] = color[0]
    green_mask[:, :, 1] = color[1]
    green_mask[:, :, 2] = color[2]

    original_image = np.array(original_image)
    other_mask = np.broadcast_to(mask == 1, original_image.shape)
    green_mask[other_mask] = 255
    
    # print("putting the layers on")
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
    print("Start Segmentation time: 0ms")
    image = Image.open(image_path)

    
    
    if image.mode == 'RGBA':
        # 转换为 RGB，去除alpha通道
        image = image.convert('RGB')
    
    # print(image.size)
    
    img_w, img_h = image.size
    back = Image.new("RGB", (img_w, img_w), "black")
    offset = ((img_w - img_w) // 2, (img_w - img_h) // 2)
    back.paste(image, offset)
    # print(back.size)
    original_image = back.copy()
    img = cv2.cvtColor(np.array(back), cv2.COLOR_RGB2BGR)
    
    
    image = back
    image = image.resize((640, 640))
    # PIL: RGB, cv2: BGR
    # Preprocess input image
    
    bound = Bound(mdla_path=mdla_path_bound)
    input_array = bound.img_preprocess(image)
    # Initialize model
    
    bound_start = time.time()
    print("Start Bound time: " + str(bound_start - start_time) + "ms")
    ret = bound.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return
    

    
    # Set input buffer for inference
    bound.SetInputBuffer(input_array, 0)
    # Execute model
    bound_mod_start = time.time()
    print("Bound model start: " + str(bound_mod_start - start_time) + "ms")
    ret = bound.Execute()
    if ret != True:
        print("Failed to Execute")
        return
    bound_mod_end = time.time()
    print("Bound model end: " + str(bound_mod_end - start_time) + "ms")
    
    # segment_img = (segment_img * 255).astype(np.uint8)
    # segment_img = (segment_img * 255)
    # image_pil = Image.fromarray(segment_img)
    # segment_img = image_pil.resize((128, 128), Image.LANCZOS)
    bound_imgs = bound.postprocess(image)
 
    img_resized = []
    for bound_img in bound_imgs:
        try:
            bound_img_resized = cv2.resize(bound_img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            img_resized.append(bound_img_resized)
        except:
            print("No leaf")
  
    
    bound_total = time.time()
    print("Finish Bound time: " + str(bound_total - start_time) + "ms")
    
    print("Bound spended time: " + str(bound_total - bound_start) + "ms")
 

    '''Segmentation'''

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
        return
    
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

    # cv2.imshow("bound_img_resized", bound_img_resized)
    # cv2.waitKey(3000)
    

    seg_total = time.time()
    
    print("Segmentation spended time: " + str(seg_total - start_time) + "ms")

    
    
    


    '''Detect Leaf Disease'''
    detect_start = time.time()
    print("Start detect time: " + str(detect_start - start_time) + "ms")
    
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
        # print(input_array.shape)
        # Set input buffer for inference
        detect.SetInputBuffer(input_array, 0)
    
        detect_mod_start = time.time()
        print("Detect model start: " + str(detect_mod_start - start_time) + "ms")
        # Execute model
        ret = detect.Execute()
        if ret != True:
            print("Failed to Execute")
            return
            
        detect_mod_end = time.time()
        print("Detect model end: " + str(detect_mod_end - start_time) + "ms")
        
        
        # print(detect.GetOutputBuffer(0))
        print(class_names[np.argmax(detect.GetOutputBuffer(0))])
        output_array.append(class_names[np.argmax(detect.GetOutputBuffer(0))])

        disease = class_names[np.argmax(detect.GetOutputBuffer(0))]
        
        detect_total = time.time()
        print("Finish Detect time: " + str(detect_total - start_time) + "ms")
        
        print("Detection spended time: " + str(detect_total - detect_start) + "ms")
        
        end_time = time.time()
        total_time = end_time - start_time
        print("Total time: " + str(total_time) + 'ms')
        
        color = (0, 0, 255)
        if(disease == "healthy"):
            color = (0,255, 0)
            
        fancy_time = time.time()
        print("Before adding fancy mask: " + str(fancy_time - start_time) + "ms")

        # print("Finished segmenting")
    try:
        original_image = original_image.resize((128 * 3, 128 * 3))
        # segment_img_with_overlay = overlay_mask_on_image(need, original_image, color)
        # print(segment_img_with_overlay.shape)
        original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        annotated_image = bound.draw_bbox_on_image(original_image, output_array)
        # annotated_image = bound.add_disease_label(annotated_image, disease, 0.9, color)
        
        cv2.imshow("Real", annotated_image)
        cv2.waitKey(3000)
        
        fancy_time_end = time.time()
        print("Finish adding fancy mask: " + str(fancy_time_end - start_time) + "ms")
    except:
        print("No image")
       
    
    # Postprocess output
    print("output_array: ")
    print(output_array)
    # detect.postprocess(image)
    
    # print("Finish annotation")
    # print(type(bound_img))

    # print(type(image_pil))
    # bound_img = image_pil.resize((128, 128), Image.LANCZOS)

    # 使用 PIL 调整图像大小
    
    # save the annotated image to the provided path 
    annotated_image = cv2.resize(annotated_image, (256, 256))
    
    # cv2.imshow("result", annotated_image)
    
    # cv2.imshow("annotation.jpg", annotated_image)
    # cv2.waitKey(1000)
    # sv_path = os.path.dirname(image_path)
    sv_path = save_path
    # print("SV PATH: " + sv_path)
    image = Image.fromarray(annotated_image)
    # print("image:" + image_path)


    # sv_path += "/annotation.jpg"
    # print(sv_path)
    image.save(sv_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Detect model with NeuronHelper')
    parser.add_argument('--dla-bound-path', type=str, default='yolo640leafdetect.mdla',
                        help='Path to the Bound mdla file')
    parser.add_argument('--dla-segment-path', type=str, default='segment_finetuned_94.mdla',
                        help='Path to the Segmentation mdla file')
    parser.add_argument('--dla-detect-path', type=str, default='det_nonseg_trans_aug_v2.mdla',
                        help='Path to the Detection mdla file')
    parser.add_argument('--image-path', type=str, default='bound_test.JPG',
                        help='Path to the input image')
    parser.add_argument('--save-path', type=str, default='final_v2.JPG',
                        help='Path to save the image')
    args = parser.parse_args()
    main(args.dla_bound_path, args.dla_segment_path, args.dla_detect_path, args.image_path, args.save_path)
