"""Implementation of the Owl client for ROS"""
import rospy
import numpy as np

from cv_bridge import CvBridge

from geometry_msgs.msg import Point           as PointMsg
from std_msgs.msg      import Int32MultiArray as Int32MultiArrayMsg

from ros_owl.srv import DetectObjects        as DetectObjectsSrv, \
                        DetectObjectsRequest as DetectObjectsRequestMsg


class OwlClient():
    """Client for the Owl detector service"""

    def __init__(self, service) -> None:
        """Initialize connection to the Owl detector service
        Args:
            service (string): Name of the service 'ros_owl' for detection
        """
        self._bridge = CvBridge()
        
        rospy.wait_for_service(f'{service}/detect')

        self._srv_owl = rospy.ServiceProxy(f'{service}/detect', DetectObjectsSrv)

    def detect(self, img_rgb, classes):
        req = DetectObjectsRequestMsg()
        req.image = self._bridge.cv2_to_imgmsg(img_rgb)
        req.classes = [str(c) for c in classes]
        res = self._srv_owl(req)
        return res.boxes, res.scores, res.labels
        '''res = self._srv_owl(
            self._bridge.cv2_to_imgmsg(img_rgb), 
            classes)
        return res.boxes, res.scores, res.labels'''