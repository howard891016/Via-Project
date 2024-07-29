#first method you call is get_bounding_box(). find_best_bounding_box() returns the bounding box that is the biggest the only argument that matters is copy thats the cropped image, detect_img is the one with the bounding boxes drawn on the original image

from NeuronRuntimeHelper import NeuronContext
from PIL import Image
import argparse
import numpy as np
import cv2


class Bound(NeuronContext):
    """
    Class YOLOv8:
    This class is used to perform object detection using the YOLOv8 model.

    Parameters
    ----------
    dla_path : str, optional
        Path to the YOLOv8 model, by default "None"
    confidence_thres : float, optional
        Confidence threshold for object detection, by default 0.5
    iou_thres : float, optional
        IOU threshold for object detection, by default 0.5
    """
    def __init__(self, mdla_path: str = "None"):
        super().__init__(mdla_path)
        """
        Initializes the YOLOv8 class.

        Parameters
        ----------
        dla_path : str
            Path to the YOLOv8 model
        confidence_thres : float
            Confidence threshold for object detection
        iou_thres : float
            IOU threshold for object detection
        """
    
    def find_best_bounding_box(self, detect_result):
        largest_area = 0
        best_bbox = None
        bestArea = 0
        for r in detect_result:
            for box in r.boxes:
                bbox = box.xywh.tolist()[0]
                # Save the x, y, width, and height to separate variables and round them to the nearest whole numbers
                x, y, w, h = map(round, bbox)  
                area = w * h
                if(area > bestArea):
                    best_bbox = box
                    bestArea = area
        if(bestArea == 0):
            return None
        else:
            return best_bbox.xywh.tolist()[0]

    def get_bounding_box(self, image, model):
        detect_img, detect_result = self.leaf_detect(image.copy(), model)
        copy = image.copy()
        best_bbox = self.find_best_bounding_box(detect_result)
        if(best_bbox != None):
            copy = np.zeros_like(image)
            x, y, w, h = map(round, best_bbox)  
            min_y =  int(round(y - 0.5 * h))
            max_y = int(round(y + 0.5 * h))
            min_x = int(round(x - 0.5 * w))
            max_x = int(round(x + 0.5 * w))
            copy[min_y:max_y, min_x:max_x] = image[min_y:max_y, min_x:max_x]
            return detect_img, copy, [min_x, min_y, w, h]
        else:
            return detect_img, copy, [0, 0, 128, 128]
        

    def img_preprocess(self, image):

        # Convert to NumPy array with the correct dtype
        dtype = np.float32
        dst_img = np.array(image, dtype=dtype)
    
        dst_img = np.expand_dims(dst_img, axis=0)  # 扩展维度添加 batch_size 维度

        return dst_img
        

    def postprocess(self, image):
        """
        Post-processing function for YOLOv8 model

        Parameters
        ----------
        image : PIL.Image
            Input image to be processed

        Returns
        -------
        None
            Function will display the result image using OpenCV
        """
        img_w, img_h = image.size
        image = np.array(image)
        bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Initilize lists to store bounding box coordinates, scores and class_ids

        cv2.imshow("result", bgr_img)
 


def main(mdla_path, image_path):
    """Main function to test YOLOv8 model using NeuronHelper

    This function tests the YOLOv8 model using NeuronHelper by:
    1. Initializing the model
    2. Loading input image
    3. Preprocessing input image
    4. Setting input buffer for inference
    5. Executing model
    6. Postprocessing output
    7. Showing result for 3 seconds
    8. Cleaning up windows
    """
    model = Bound(mdla_path=mdla_path)

    # Initialize model
    ret = model.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return

    
    # Load input image
    # image = Image.open(image_path)
    image = cv2.imread(image_path)
    # # if(image.shape == (1, 128, 128, 3)):
    # #     image = image[0]
    # # else:
    # # image = cv2.resize(image, (128, 128))
    image = cv2.resize(image, (128, 128))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_array = model.img_preprocess(image)
 

    # Preprocess input image
    # input_array = image

    # Set input buffer for inference
    model.SetInputBuffer(input_array, 0)
    # print(input_array.shape)

    # Execute model
    ret = model.Execute()
    if ret != True:
        print("Failed to Execute")
        return
    
    detect_result = model.GetOutputBuffer(0)
    # print(detect_result.shape)
    result = detect_result.reshape((5, 336))
    # print(result.shape)
    # print(result)
    bounding_boxes = []

    for i in range(result.shape[1]):
        
        x_center = result[0, i] * 128
        y_center = result[1, i] * 128
        width = int(round(result[2, i] * 128))
        height = int(round(result[3, i] * 128))
        confidence = result[4, i]

        print(f"X_center: {x_center}")
        print(f"Y_center: {y_center}")
        print(f"width: {width}")
        print(f"height: {height}")
        print(f"confidence: {confidence}")
        

        # 计算左上角和右下角的坐标
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)
        
        bounding_boxes.append([x_min, y_min, x_max, y_max, confidence])
        

    # print(len(bounding_boxes))
    confidence_threshold = 0.5
    filtered_boxes = [box for box in bounding_boxes if box[4] > confidence_threshold]
    # for box in filtered_boxes:
        
    best_area = -1
    best = None

    for box in filtered_boxes:
        x_min, y_min, x_max, y_max, confidence = box
    
        # print(f"X_min: {x_min}")
        # print(f"Y_min: {y_min}")
        # print(f"X_max: {x_max}")
        # print(f"Y_max: {y_max}")

        width = x_max - x_min
        height = y_max - y_min

        area = width * height
        if area > best_area:
            best_area = area
            best = x_min, y_min, x_max, y_max, confidence
    
    if best != None:
        x_min, y_min, x_max, y_max, confidence = best
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 255), 4)
        
            # x_min, y_min, x_max, y_max, confidence = box
            # cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 255), 4)
        text_x = max(int(x_min), 0)
        text_y = max(int(y_min) - 10, 0)
        # label = f'Confidence: {confidence:.2f}'
        # cv2.putText(image, str(label), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.putText(image, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
 

        cv2.imshow('Detected Image', image)




    # copy = input_array.copy()
    # best_bbox = model.find_best_bounding_box(detect_result)
    # if(best_bbox != None):
    #     copy = np.zeros_like(image)
    #     x, y, w, h = map(round, best_bbox)  
    #     min_y =  int(round(y - 0.5 * h))
    #     max_y = int(round(y + 0.5 * h))
    #     min_x = int(round(x - 0.5 * w))
    #     max_x = int(round(x + 0.5 * w))
    #     copy[min_y:max_y, min_x:max_x] = image[min_y:max_y, min_x:max_x]

    #     print(copy)
        
    # detect_result = detect_result[0].ravel()
    # detect_img = detect_result[0].plot()
    # detect_img = detect_result.plot()
    # detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)    

    
    cv2.waitKey(0)

    # Clean up windows
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Detect model with NeuronHelper')
    parser.add_argument('--dla-path', type=str, default='best_float32.mdla',
                        help='Path to the Detection mdla file')
    parser.add_argument('--image-path', type=str, default='transforma_test.jpg',
                        help='Path to the input image')
    args = parser.parse_args()

    main(args.dla_path, args.image_path)
