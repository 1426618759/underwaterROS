#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2-7B-Instruct"
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto"), "./qwen_finetuned")  # 假设微调模型在本地
tokenizer = AutoTokenizer.from_pretrained(model_name)

def callback(msg):
    prompt = msg.data
    messages = [{"role": "system", "content": "解析为 JSON 指令: action (e.g., patrol), params (e.g., area: A)."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_new_tokens=128, temperature=0.1)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    try:
        command = json.loads(response)
        cmd = Twist()
        if command["action"] == "forward":
            cmd.linear.x = command.get("speed", 0.5)
        pub.publish(cmd)
    except json.JSONDecodeError:
        rospy.logerr("Invalid JSON from Qwen")

if __name__ == "__main__":
    rospy.init_node("nlp_auv_control")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rospy.Subscriber("/nl_command", String, callback)
    rospy.spin()
