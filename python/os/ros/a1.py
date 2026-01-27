#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import datetime

def extract_images_from_bag(bag_file, output_parent_dir, topic_name):
    """从单个bag文件中提取图像并保存"""
    bag_name = os.path.splitext(os.path.basename(bag_file))[0]
    output_dir = os.path.join(output_parent_dir, bag_name)
    
    # 2. 创建多级输出目录（支持子目录）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # exist_ok=True：若目录已存在不报错
        print(f"创建多级目录: {output_dir}")
    
    bridge = CvBridge()
    count = 0
    processed_count = 0
    
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                try:
                    # 3. 处理NV21格式和其他格式的图像转换
                    encoding = msg.encoding
                    width = msg.width
                    height = msg.height
                    
                    if encoding == 'yuvnv21':
                        nv21_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((height * 3 // 2, width))
                        cv_image = cv2.cvtColor(nv21_data, cv2.COLOR_YUV2BGR_NV21)
                    else:
                        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    
                    # 4. 每10张保存1张（按你的需求保留）
                    if processed_count % 1 == 0:
                        timestamp = datetime.datetime.fromtimestamp(t.to_sec()).strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        filename = f"{count:05d}_{timestamp}.png"
                        filepath = os.path.join(output_dir, filename)
                        cv2.imwrite(filepath, cv_image)
                        print(f"保存图像: {filepath}")
                        count += 1
                    
                    processed_count += 1
                    
                except CvBridgeError as e:
                    print(f"CV桥接错误: {e}")
                except Exception as e:
                    print(f"处理图像时出错: {e}")
        
        print(f"完成处理 {bag_file}! 共处理了 {processed_count} 张图片，保存了 {count} 张到 {output_dir}")
        return count
        
    except Exception as e:
        print(f"处理bag文件 {bag_file} 时出错: {e}")
        return 0

def extract_images_from_dir():
    # 配置信息：确保base_bag_dir是原始bag的根目录（所有多级子目录都在其下）
    base_bag_dir = "/home/dreame/morgan/file/test/121/喷头/"  # 原始bag根目录（多级目录的起点）
    topic_name = "/camera/ai"  # 图像话题名称
    
    # 检查原始根目录是否存在
    if not os.path.exists(base_bag_dir):
        print(f"错误: 原始bag根目录 {base_bag_dir} 不存在!")
        return
    
    #
    
    # 3. 递归遍历所有多级子目录，收集所有.bag文件
    bag_files = []
    for root, dirs, files in os.walk(base_bag_dir):  # os.walk：递归遍历所有子目录
        for file in files:
            if file.lower().endswith('.bag'):
                bag_file_path = os.path.join(root, file)
                bag_files.append(bag_file_path)
    
    # 检查是否找到bag文件
    if not bag_files:
        print(f"在 {base_bag_dir} 及其所有子目录中未找到任何bag文件")
        return
    
    print(f"找到 {len(bag_files)} 个bag文件（含多级子目录），开始处理...")
    
    # 处理每个bag文件
    total_saved = 0
    for bag_file in bag_files:
        print(f"\n开始处理: {bag_file}")
        saved = extract_images_from_bag(bag_file, base_bag_dir, topic_name)
        total_saved += saved
    
    print(f"\n所有处理完成! 总共保存了 {total_saved} 张图像到 {base_bag_dir}（含多级子目录）")

if __name__ == "__main__":
    extract_images_from_dir()