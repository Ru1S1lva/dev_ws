from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

class yolo_pose(Node):

    def __init__(self):
        super().__init__('Yolo_Obj')

        self.model = YOLO('/home/ruisilva/dev_ws/src/yolo_pose/yolo_pose/yolov8n-pose.pt')

        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        #self.subscription 
        self.img_pub = self.create_publisher(Image, "/yolo_pose", 1)


    def camera_callback(self, data):

        img = bridge.imgmsg_to_cv2(data, "bgr8")
        results = self.model.track(img, show=True, tracker="bytetrack.yaml")  # with ByteTrack

        format_detected = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
                c = int(box.cls[0].item()) # Class ID
                self.get_logger().info(f'C: {c}')

                top, left, bottom, right = b
                width = right - left
                height = bottom - top
                area = width * height

                #print area
                #self.get_logger().info(f'Area: {area}') linha comandos
                #text = f'ID{idx}: {area}'
                #cv2.putText(img, text, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        annotated_frame = results[0].plot()
        img_msg = bridge.cv2_to_imgmsg(annotated_frame)  

        self.img_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    yolo_pose_node = yolo_pose()
    rclpy.spin(yolo_pose_node)
    yolo_pose_node.destroy_node()
    rclpy.shutdown()
