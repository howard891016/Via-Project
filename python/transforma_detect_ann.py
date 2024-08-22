from NeuronRuntimeHelper import NeuronContext
from PIL import Image
import argparse
import numpy as np
import cv2
import time

class Detect(NeuronContext):
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

    def img_preprocess(self, image):

        # Convert to NumPy array with the correct dtype
        dtype = np.float32
        dst_img = np.array(image, dtype=dtype)
    
        # 如果模型要求输入是四维的，例如 U-Net 等模型通常要求输入是 (batch_size, height, width, channels)
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

        # cv2.imshow("result", bgr_img)
        

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
    start_time = time.time()
    model = Detect(mdla_path=mdla_path)

    # Initialize model
    ret = model.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 读取的图像是 BGR 格式，转换为 RGB 格式
    
    # Load input image
    image = Image.open(image_path)
    image = image.resize((128, 128))

    # Preprocess input image
    input_array = model.img_preprocess(image)

    # Set input buffer for inference
    model.SetInputBuffer(input_array, 0)

    # Execute model
    ret = model.Execute()
    if ret != True:
        print("Failed to Execute")
        return
    
    output_array = []

    # Postprocess output
    class_names = ['blight','citrus' ,'healthy', 'measles', 'mildew', 'mite', 'mold', 'rot', 'rust', 'scab', 'scorch', 'spot', 'virus']
    print(model.GetOutputBuffer(0))
    print(class_names[np.argmax(model.GetOutputBuffer(0))])
    model.postprocess(image)
    end_time = time.time()

    cv2.waitKey(3000)

    # Clean up windows
    cv2.destroyAllWindows()
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
