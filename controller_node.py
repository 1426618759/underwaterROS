#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

def callback(msg):
    cmd = Twist()
    if msg.data == "patrol":
        cmd.linear.x = 0.2  # 前进
        cmd.angular.z = 0.1  # 转弯
    pub.publish(cmd)

if __name__ == "__main__":
    rospy.init_node("auv_controller")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rospy.Subscriber("/auv_control", String, callback)
    rospy.spin()
