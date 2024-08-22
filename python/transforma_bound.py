#first method you call is get_bounding_box(). find_best_bounding_box() returns the bounding box that is the biggest the only argument that matters is copy thats the cropped image, detect_img is the one with the bounding boxes drawn on the original image

from NeuronRuntimeHelper import NeuronContext
from PIL import Image, ImageOps
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
    def __init__(self, mdla_path: str = "None", confidence_thres=0.5, iou_thres=0.5):
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
        self.confidence_thres = confidence_thres
        """Confidence threshold for object detection"""
        self.iou_thres = iou_thres
        """IOU threshold for object detection"""

    def draw_boxes(self, image, box, score, class_id, color=(0, 255, 0)):
        """Draws a bounding box on the image based on the detected class and confidence

        Parameters
        ----------
        image : numpy.ndarray
            Image to draw bounding box on
        box : tuple
            (x, y, width, height) of the bounding box
        score : float
            Confidence of the detected class
        class_id : int
            Class ID of detected object
        """
        # Convert the box coordinates to integers
        x1, y1, w, h = [int(v) for v in box]
        # Set the color for the bounding box

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color, 2)
        # Define the label text
        label = f"{class_id}: {score:.2f}"  # class_id: confidence
        # Calculate the size of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Determine the position of the label on the image
        label_x = x1
        # label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        label_y = y1
        # Add a rectangular background to the label
        cv2.rectangle(
            image,
            (int(label_x), int(label_y)),
            (int(label_x), int(label_y)),
            # (int(label_x), int(label_y - label_height)),
            # (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )
        # Add the label text
        # cv2.putText(
        #     image,
        #     label,
        #     (int(label_x), int(label_y)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,  # font scale
        #     (0, 0, 0),  # text color
        #     1,  # line thickness
        #     cv2.LINE_AA,
        # )
    
        

    def img_preprocess(self, image):

        # Convert to NumPy array with the correct dtype
        dtype = np.float32
        dst_img = np.array(image, dtype=dtype)
        dst_img /= 255.0
        dst_img = np.expand_dims(dst_img, axis=0)  # 扩展维度添加 batch_size 维度

        return dst_img
    def add_disease_label(self, image, prediction, confidence, color):
        # print("ADDing diseaes label")
        boxes = self.get_box()
        label_image = image.copy()
        for box in boxes:
            x, y, w, h = [int(v) for v in box]
            label_text = f'{prediction} ({confidence:.2f})'
            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 0.4
            thickness = 1

            # print("Using width and height")
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            # print("Got width and height")
            # cv2.rectangle(label_image, (x+3, y), (x + text_width + 8, y + baseline + 10), (255, 255, 255), cv2.FILLED)
            # print("Putting text")
            # cv2.putText(label_image, label_text, (x + 5,y + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color, 1, 1)
        return label_image  
    def get_box(self):
        # print("CALLING")
        img_w = 128
        img_h = 128
        # print("GEttting the prediction")
        output = self.GetOutputBuffer(0)
        # print("GOt the prediction")
        # Initilize lists to store bounding box coordinates, scores and class_ids

        boxes = []
        scores = []
        class_ids = []

        for pred in output:
            # Transpose the output from (24, 8400) to (8400, 24)
            pred = np.transpose(pred)
            for box in pred:
                # Get bounding box coordinates, scaled by image width and height
                x, y, w, h = box[:4]
                x = int(x * img_w)
                y = int(y * img_h)
                w = int(w * img_w)
                h = int(h * img_h)

                # Calculate center coordinates of the bounding box
                x1 = x - w / 2
                y1 = y - h / 2

                # Append the bounding box coordinates, scores and class_ids to their respective lists
                boxes.append([x1, y1, w, h])
                idx = np.argmax(box[4:])
                scores.append(box[idx + 4])
                class_ids.append(idx)

        # Filter out low confidence bounding boxes using non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_thres, self.iou_thres
        )
        
        if len(indices) == 0:
            return None

        final_scores = [scores[i] for i in indices]
        final_boxes = [boxes[i] for i in indices]
        final_class_ids = [class_ids[i] for i in indices]

        max_score = max(final_scores)
        max_indices = np.where(final_scores == max_score)[0]
        max_index = max_indices[0]
        # Get the bounding box coordinates, score and class_id for the selected bounding boxes
        box = final_boxes[max_index]
        return final_boxes
        
    def draw_bbox_on_image(self, output_image, output_array):
        # print("CALLING")
        img_w = 128 * 3
        img_h = 128 * 3
        # print("GEttting the prediction")
        output = self.GetOutputBuffer(0)
        # print("GOt the prediction")
        # Initilize lists to store bounding box coordinates, scores and class_ids

        boxes = []
        scores = []
        class_ids = []

        for pred in output:
            # Transpose the output from (24, 8400) to (8400, 24)
            pred = np.transpose(pred)
            for box in pred:
                # Get bounding box coordinates, scaled by image width and height
                x, y, w, h = box[:4]
                x = int(x * img_w)
                y = int(y * img_h)
                w = int(w * img_w)
                h = int(h * img_h)

                # Calculate center coordinates of the bounding box
                x1 = x - w / 2
                y1 = y - h / 2

                # Append the bounding box coordinates, scores and class_ids to their respective lists
                boxes.append([x1, y1, w, h])
                idx = np.argmax(box[4:])
                scores.append(box[idx + 4])
                class_ids.append(idx)

        # Filter out low confidence bounding boxes using non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_thres, self.iou_thres
        )
        
        if len(indices) == 0:
            return None
        
        cur = 0
        
        for i in indices:
            x, y, w, h = boxes[i]
            x = (int(x) if int(x) > 0 else 0)
            y = (int(y) if int(y) > 0 else 0)
            w = (int(w) if int(w) > 0 else 0)
            h = (int(h) if int(h) > 0 else 0)
            
            disease = output_array[cur]
            cur = cur + 1
            
            color = (0, 0, 255)
            if(disease == "healthy"):
                color = (0,255, 0)
      
            if(w * h > (img_h * img_w * 0.0278)):
                self.draw_boxes(output_image, boxes[i], scores[i], class_ids[i], color)
        # print("Done drawing boxes")
        return output_image
        
    def resize_and_pad(self, image, side):
        
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_w, img_h = image_pil.size
        print("img_w, img_h: " + str(img_w) + ", " + str(img_h))
        back = Image.new("RGB", (side, side), "black")
        offset = ((side - img_w) // 2, (side - img_h) // 2)
        back.paste(image_pil, offset)
        back = cv2.cvtColor(np.array(back), cv2.COLOR_RGB2BGR)

        return back
        
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
        output = self.GetOutputBuffer(0)
        # Initilize lists to store bounding box coordinates, scores and class_ids

        boxes = []
        scores = []
        class_ids = []

        for pred in output:
            # Transpose the output from (24, 8400) to (8400, 24)
            pred = np.transpose(pred)
            for box in pred:
                # Get bounding box coordinates, scaled by image width and height
                x, y, w, h = box[:4]
                x = int(x * img_w)
                y = int(y * img_h)
                w = int(w * img_w)
                h = int(h * img_h)

                # Calculate center coordinates of the bounding box
                x1 = x - w / 2
                y1 = y - h / 2

                # Append the bounding box coordinates, scores and class_ids to their respective lists
                boxes.append([x1, y1, w, h])
                idx = np.argmax(box[4:])
                scores.append(box[idx + 4])
                class_ids.append(idx)

        # Filter out low confidence bounding boxes using non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_thres, self.iou_thres
        )
        
        if len(indices) == 0:
            return None
        
        crop_images = []
        
        for i in indices:
            x, y, w, h = boxes[i]
            x = (int(x) if int(x) > 0 else 0)
            y = (int(y) if int(y) > 0 else 0)
            w = (int(w) if int(w) > 0 else 0)
            h = (int(h) if int(h) > 0 else 0)
            boxes[i] = x, y, w, h
            # self.draw_boxes(bgr_img, boxes[i], scores[i], class_ids[i])
            
            print("x: " + str(x) + ", y: " + str(y) + ", w: " + str(w) + ", h: " + str(h))
            mask = np.zeros((img_w, img_h), dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            # new_img = bgr_img
            # result = cv2.bitwise_and(new_img, new_img, mask=mask)
            
            # cv2.imshow("new_img", new_img)
            # cv2.waitKey(3000)
            
            
            cropped_image = bgr_img[y:y+h, x:x+w]
            if(w != h):
                if(w > h): 
                    cropped_image = self.resize_and_pad(cropped_image, w)
                else:
                    cropped_image = self.resize_and_pad(cropped_image, h)
            
            if(w * h < (img_h * img_w * 0.0278)):
                print("img_h: " + str(img_h) + ", img_w: " + str(img_w))
                print("w: " + str(w) + ", h: " + str(h))
                print('Detect too small leaf')
            else:
                # cv2.imshow("cropped_image", cropped_image)
                # cv2.waitKey(2000)
                crop_images.append(cropped_image)
            print("crop images: ")
            print(len(crop_images))
        
            # result = bgr_img
        
        
        
        
        # cv2.imshow("cropped_image", cropped_image)
        # cv2.waitKey(3000)
        return crop_images


def main(mdla_path, image_path):

    model = Bound(mdla_path=mdla_path)

    # Initialize model
    ret = model.Initialize()
    if ret != True:
        print("Failed to initialize model")
        return

    
    image = Image.open(image_path)
    image = image.resize((128, 128))

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_array = model.img_preprocess(image)
 

    # Set input buffer for inference
    model.SetInputBuffer(input_array, 0)
    # print(input_array.shape)

    # Execute model
    ret = model.Execute()
    if ret != True:
        print("Failed to Execute")
        return
    

    model.postprocess(image)

   
    cv2.waitKey(3000)

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
