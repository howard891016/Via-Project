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

def draw_bbox_on_image(output_image, output_array, bound_output):
        img_w = 128 * 3
        img_h = 128 * 3
        # print("GEttting the prediction")
        output = bound_output
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
            boxes, scores, 0.5, 0.5
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
                draw_boxes(output_image, boxes[i], scores[i], class_ids[i], color)
        # print("Done drawing boxes")
        return output_image

def annotation_process(save_path, output_array, bound_output):
    try:
        original_image = original_image.resize((128 * 3, 128 * 3))
        # segment_img_with_overlay = overlay_mask_on_image(need, original_image, color)
        original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        annotated_image = draw_bbox_on_image(original_image, output_array, bound_output)
        # annotated_image = bound.add_disease_label(annotated_image, disease, 0.9, color)
        
        cv2.imshow("Real", annotated_image)
        cv2.waitKey(3000)
        
    except:
        print("No image")
       
    
    # Postprocess output
    print("output_array: ")
    print(output_array)
    

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
