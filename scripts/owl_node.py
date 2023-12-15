#!/usr/bin/env python

import numpy as np
import rospy
import cv2

from std_msgs.msg import Float32MultiArray as Float32MultiArrayMsg, MultiArrayDimension

from cv_bridge   import CvBridge

from ros_owl.srv import DetectObjects        as DetectObjectsSrv, \
                        DetectObjectsRequest as DetectObjectsRequestMsg, \
                        DetectObjectsResponse as DetectObjectsResponseMsg

from ros_owl import Owl
from PIL import Image


if __name__ == '__main__':
    rospy.init_node('ros_owl')

    #model = rospy.get_param('~model', 'vit_l')
    #cuda  = rospy.get_param('~cuda', 'cuda')

    bridge = CvBridge()

    print('Starting Owl...')

    owl = Owl()

    def bounding_boxes_to_float32multiarray(bounding_boxes):
        # Flatten the list of bounding boxes into a single list
        dims = [MultiArrayDimension(size=len(bounding_boxes[0]), stride=len(bounding_boxes[0])*len(bounding_boxes), label='box') for _ in bounding_boxes]

        flat_list = [item for sublist in bounding_boxes for item in sublist]

        multiarray = Float32MultiArrayMsg()
        multiarray.layout.dim = dims
        multiarray.layout.data_offset = 0
        multiarray.data = flat_list

        return multiarray
    
    def srv_detection(req : DetectObjectsRequestMsg):
        try:
            img = cv2.cvtColor(bridge.imgmsg_to_cv2(req.image), cv2.COLOR_BGR2RGB)
            #classes = np.asarray(req.classes)
            classes = req.classes

            print("Detecting objects:")
            print(img.shape)

            boxes, scores, labels = owl.detect(Image.fromarray(img), classes)

            boxes_msg = bounding_boxes_to_float32multiarray(boxes)

            res = DetectObjectsResponseMsg()
            res.boxes  = boxes_msg
            res.scores = scores.tolist()
            res.labels = labels.tolist()
            return res
        
        except Exception as e:
            print(f'{e}')
            raise Exception('Failure during service call. Check output on Owl node.')

    srv = rospy.Service('~detect', DetectObjectsSrv, srv_detection)

    print('Owl is ready')

    while not rospy.is_shutdown():
        rospy.sleep(0.2)
    