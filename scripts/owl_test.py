#!/usr/bin/env python
import cv2
import rospy
import numpy as np

from cv_bridge import CvBridge
from pathlib   import Path

from geometry_msgs.msg import Point as PointMsg
from std_msgs.msg      import Int32MultiArray as Int32MultiArrayMsg
from ros_owl.srv import DetectObjects        as DetectObjectsSrv, \
                        DetectObjectsRequest  as DetectObjectsRequesttMsg

from ros_owl import OwlClient
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers.image_utils import ImageFeatureExtractionMixin
mixin = ImageFeatureExtractionMixin()

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ImageMsg
from PIL import Image

def plot_predictions(image, text_queries, scores, boxes, labels):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for box_index in range(num_boxes):
        if scores[box_index] < score_threshold:
            continue
        start_index = box_index * box_size
        end_index = start_index + box_size
        box = boxes.data[start_index:end_index]

        cx, cy, w, h = box
        ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
                [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
        ax.text(
            cx - w / 2,
            cy + h / 2 + 0.015,
            f"{text_queries[0]}: {scores[box_index]:1.2f}",
            ha="left",
            va="top",
            color="red",
            bbox={
                "facecolor": "white",
                "edgecolor": "red",
                "boxstyle": "square,pad=.3"
            })
    
    plt.show()
        
if __name__ == '__main__':
    rospy.init_node('ros_owl_test')

    print('Waiting for Owl service...')
    rospy.wait_for_service('ros_owl/detect')
    print('Found Owl service')

    #image  = cv2.imread(f'{Path(__file__).parent}/../data/orange_cup.png')
    #image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bridge = CvBridge()
    image = rospy.wait_for_message('/rgb/image_rect_color', ImageMsg)
    image = cv2.cvtColor(bridge.imgmsg_to_cv2(image), cv2.COLOR_BGR2RGB)
    w, h = image.shape[1], image.shape[0]
    print(w, h)
    #im = Image.fromarray(image)
    #im.save("/home/karthikm/ws_grasp/src/ros_owl/scripts/your_file.jpeg")
    
    owl = OwlClient('ros_owl')

    texts = [["a photo of a cup"]]
    i = 0
    text = texts[i]

    boxes, scores, labels = owl.detect(image, texts)
    print(boxes)

    box_size = 4  # The size of each bounding box
    num_boxes = len(boxes.data) // box_size
    score_threshold = 0.1

    #fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    #ax.imshow(image)
    #ax.set_axis_off()

    for box_index in range(num_boxes):
        if scores[box_index] < score_threshold:
            continue
        start_index = box_index * box_size
        end_index = start_index + box_size
        box = boxes.data[start_index:end_index]
        box = [int(x) for x in box]
        #box[1] = h - box[1]
        #box[3] = h - box[3]
        #box[1], box[3] = box[3], box[1]

        # Convert the coordinates to integers
        image = cv2.rectangle(image, box[:2], box[2:], (255,0,0), 5)

        #rect = patches.Rectangle((top_left_x, top_left_y + height), width, height, linewidth=1, edgecolor='r', facecolor='none')
        #plt.gca().add_patch(rect)

        # Draw the bounding box on the image
        #rect = patches.Rectangle(top_left, bottom_right[0]-top_left[0], bottom_right[1]-top_left[1], linewidth=1, edgecolor='r', facecolor='none')

        #print(f"Bounding Box {box_index + 1}: {single_box}")
        print(f"Detected {text[labels[box_index]]} with confidence {round(scores[box_index], 3)} at location {box}")

    #plot_predictions(image, text, scores, boxes, labels)

    plt.imshow(image)   
    plt.show()
#plt.imshow(np.flipud(image), origin='lower')
#plt.show()