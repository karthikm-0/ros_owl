#!/usr/bin/env python
import cv2
import rospy
import numpy as np

from cv_bridge import CvBridge
from pathlib   import Path

from geometry_msgs.msg import Point as PointMsg
from std_msgs.msg      import Int32MultiArray as Int32MultiArrayMsg

from ros_owl import OwlClient
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers.image_utils import ImageFeatureExtractionMixin
mixin = ImageFeatureExtractionMixin()

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ImageMsg
from PIL import ImageDraw, Image

# TODO: Hack that needs fixing
np.float = np.float64  # temp fix for following import
import ros_numpy

if __name__ == '__main__':
    rospy.init_node('ros_owl_test')

    print('Waiting for Owl service...')
    rospy.wait_for_service('ros_owl/detect')
    print('Found Owl service')

    bridge = CvBridge()
    image = rospy.wait_for_message('/rgb/image_rect_color', ImageMsg)
    image = bridge.imgmsg_to_cv2(image)
    plt.imshow(image)
    w, h = image.shape[1], image.shape[0]
    print(w, h)
    
    owl = OwlClient('ros_owl')

    texts = [["a photo of a tape"]]
    i = 0
    text = texts[i]

    boxes, scores, labels, output_image = owl.detect(image, texts)
    print(boxes)

    box_size = 4  # The size of each bounding box
    num_boxes = len(boxes.data) // box_size
    score_threshold = 0.1

    visualized_image = ros_numpy.numpify(output_image)
    visualized_image = Image.fromarray(visualized_image)
    draw = ImageDraw.Draw(visualized_image)

    for box_index in range(num_boxes):
        if scores[box_index] < score_threshold:
            continue
        start_index = box_index * box_size
        end_index = start_index + box_size
        box = boxes.data[start_index:end_index]
        box = [int(x) for x in box]

        # Convert the coordinates to integers
        draw.rectangle(xy=((box[0], box[1]), (box[2], box[3])), outline="red")
        draw.text(xy=(box[0], box[1]), text=text)

        #print(f"Bounding Box {box_index + 1}: {single_box}")
        print(f"Detected {text[labels[box_index]]} with confidence {round(scores[box_index], 3)} at location {box}")

    visualized_image.show()
