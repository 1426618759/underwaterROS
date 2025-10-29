#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from transformers import CLIPProcessor, CLIPModel
import torch
from std_msgs.msg import String

bridge = CvBridge()
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def callback(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    inputs = processor(text=["shipwreck", "ocean"], images=cv_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    if probs[0][0] > 0.8:
        result_pub.publish(String("Task verified: Shipwreck detected"))
        rospy.loginfo("Shipwreck detected")

if __name__ == "__main__":
    rospy.init_node("visual_feedback")
    result_pub = rospy.Publisher("/visual_result", String, queue_size=10)
    rospy.Subscriber("/camera/image_raw", Image, callback)
    rospy.spin()
