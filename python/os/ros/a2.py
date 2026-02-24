import rosbag
from sensor_msgs.msg import Image
import numpy as np
import cv2
import os
import argparse

def split_stereo_image(msg, width, height):
   
    Y_SIZE = width * height  # Y分量大小
    UV_SIZE = width * height // 2  # UV分量大小（NV21中U和V共占1/2）
    SINGLE_SIZE = Y_SIZE + UV_SIZE  # 单目图像总大小（Y+UV）
    STEREO_SIZE = 2 * SINGLE_SIZE  # 双目图像总大小（左+右）

    # 大小校验
    if len(msg.data) != STEREO_SIZE:
        raise ValueError(f"图像数据大小不匹配: 期望 {STEREO_SIZE}, 实际 {len(msg.data)}")

    src = np.frombuffer(msg.data, dtype=np.uint8)  # 转换为numpy数组

    # 1. 构造左目YUV NV21图像
    left_yuv = np.empty((height * 3 // 2, width), dtype=np.uint8)
    # 拷贝Y分量
    left_yuv[:height, :] = src[:Y_SIZE].reshape(height, width)
    # 拷贝UV分量（NV21格式：V在前，U在后）
    left_yuv[height:, :] = src[Y_SIZE : Y_SIZE + UV_SIZE].reshape(height//2, width)

    # 2. 构造右目YUV NV21图像
    right_yuv = np.empty((height * 3 // 2, width), dtype=np.uint8)
    # 拷贝Y分量
    right_yuv[:height, :] = src[SINGLE_SIZE : SINGLE_SIZE + Y_SIZE].reshape(height, width)
    # 拷贝UV分量
    right_yuv[height:, :] = src[SINGLE_SIZE + Y_SIZE : STEREO_SIZE].reshape(height//2, width)

    # 3. 转换为BGR格式
    left_bgr = cv2.cvtColor(left_yuv, cv2.COLOR_YUV2BGR_NV21)
    right_bgr = cv2.cvtColor(right_yuv, cv2.COLOR_YUV2BGR_NV21)

    return left_bgr, right_bgr

def extract_stereo_images(bag_path, output_dir, width, height):
   
    # 创建输出文件夹
    left_dir = os.path.join(output_dir, "left")
    right_dir = os.path.join(output_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    # 初始化计数器
    img_count = 0

    try:
        # 打开rosbag
        with rosbag.Bag(bag_path, 'r') as bag:
            # 遍历/cam/dual话题的所有消息
            for topic, msg, t in bag.read_messages(topics=['/cam_dual']):
                try:
                    # 拆分左右目图像
                    left_img, right_img = split_stereo_image(msg, width, height)
                    
                    # 生成保存路径（6位序号，如000000）
                    left_path = os.path.join(left_dir, f"{img_count:06d}.jpg")
                    right_path = os.path.join(right_dir, f"{img_count:06d}.jpg")
                    
                    # 保存图像
                    cv2.imwrite(left_path, left_img)
                    cv2.imwrite(right_path, right_img)
                    
                    # 每处理100张图像打印一次进度
                    if img_count % 1 == 0 and img_count != 0:
                        print(f"已处理 {img_count} 张图像")
                    
                    img_count += 1

                except ValueError as e:
                    print(f"处理第 {img_count} 张图像失败: {e}，已跳过")
                except Exception as e:
                    print(f"处理第 {img_count} 张图像时发生未知错误: {e}，已跳过")

        print(f"处理完成！共提取 {img_count} 对图像")
        print(f"左目图像保存至: {left_dir}")
        print(f"右目图像保存至: {right_dir}")

    except FileNotFoundError:
        print(f"错误：未找到rosbag文件 {bag_path}")
    except Exception as e:
        print(f"处理rosbag时发生错误: {e}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从rosbag中提取双目图像（/cam/dual话题）')
    parser.add_argument('--bag_file', default='/project/data/1209/2568/2541-20251121-4.bag',help='输入的rosbag文件路径')
    parser.add_argument('--output_dir', default='/project/data/1209/2568/2541-20251121-4',help='图像保存的根目录')
    parser.add_argument('--width', type=int, default=640, help='单目图像宽度（像素）')
    parser.add_argument('--height', type=int, default=360, help='单目图像高度（像素）')
    
    args = parser.parse_args()
    
    # 调用提取函数
    extract_stereo_images(args.bag_file, args.output_dir, args.width, args.height)