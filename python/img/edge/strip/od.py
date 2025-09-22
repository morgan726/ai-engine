import cv2
import numpy as np
from ultralytics import YOLO
import os
from esg import get_subpixel_corners_near_vertices, process_gray_image
model = YOLO("/home/dreame/dmt/data/nvme7/alg/cv/detection/ultralytics/runs/train/exp17/weights/best.pt")  
checkpoint = "/home/dreame/dmt/data/nvme7/code/toolbox/mm/mmpose/work_dirs/rtmpose-m-corner/best_PCK_epoch_150.pth"
input_folder = "/home/dreame/dmt/program/project/apriltagone-MOVA3000_two_rect/data/"  # 输入图像文件夹
# input_folder = "/home/dreame/dmt/program/data/dataset-0918"
output_folder = "/home/dreame/dmt/data/nvme7/dataset/pose/strip/result1"  # 可视化结果文件夹
output_folder2 = "/home/dreame/dmt/data/nvme7/dataset/pose/strip/result2" 
config = "/home/dreame/dmt/data/nvme7/code/toolbox/mm/mmpose/data/rtmpose-m-corner.py"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)
from mmpose.apis import init_model, inference_topdown

model_pose = init_model(config, checkpoint, "cuda")
model_pose.eval()

def get_centermost_boxes(boxes, img_width, img_height, keep=2):
    """
    计算边界框中心到图像中心的距离，保留距离最近的N个框
    
    :param boxes: 边界框列表 (x1, y1, x2, y2)
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :param keep: 保留的框数量
    :return: 过滤后的边界框
    """
    # 图像中心点坐标
    img_center = (img_width / 2, img_height / 2)
    
    # 计算每个框的中心和到图像中心的距离
    box_info = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2, _, _ = box
        # 计算边界框中心点
        box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        # 计算欧氏距离（简化为平方距离，避免开方运算）
        distance = (box_center[0] - img_center[0])**2 + (box_center[1] - img_center[1])** 2
        box_info.append((distance, idx))  # (距离, 原始索引)
    
    # 按距离排序（升序，距离越小越居中）
    box_info.sort()
    
    # 取前N个最居中的框
    keep_indices = [info[1] for info in box_info[:keep]]
    filtered_boxes = [boxes[i] for i in keep_indices]

    bboxs = []
    for box in filtered_boxes:
        x1, y1, x2, y2, _, _ = box  
        corners_vertices = [
            (int(x1), int(y1)),  # 左上角，转换为 int
            (int(x2), int(y1)),  # 右上角，转换为 int
            (int(x2), int(y2)),  # 右下角，转换为 int
            (int(x1), int(y2))   # 左下角，转换为 int
        ]

        bboxs.append(corners_vertices)
    
    return bboxs

def has_duplicate_keypoints(keypoints, threshold=3.0):
    """
    判断是否存在重复的关键点
    
    参数:
        keypoints: 关键点数组，形状为(N, 2)
        threshold: 距离阈值，小于该值的点被认为是重复的
        
    返回:
        bool: 存在重复关键点返回True，否则返回False
    """
    n = len(keypoints)
    if n < 2:  # 少于2个点，不可能有重复
        return False
        
    # 计算所有点之间的距离
    for i in range(n):
        for j in range(i + 1, n):
            # 计算欧氏距离
            distance = np.sqrt(np.sum((keypoints[i] - keypoints[j]) **2))
            
            if distance < threshold:
                return True  # 发现重复点，立即返回True
                
    return False  # 没有发现重复点


i = 0
sub = 0
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        print(image_path)
        image = cv2.imread(image_path)
        output_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_height, img_width = image.shape[:2]
        if image is None:
            print("无法读取图像，请检查路径！")
            exit()

        # 执行目标检测
        results = model(image)[0]  # 获取检测结果
        objs = get_centermost_boxes(results.boxes.data.tolist(),img_width,img_height)
        for box in objs:
            x1, y1 = box[0]
            x2, y2 = box[2] if len(box) > 2 else box[1]
            
            i = i + 1

            save_path = os.path.join(output_folder, f"{i}_strip_0818.jpg")
            # cv2.imwrite(save_path,image[y1-sub:y2+sub, x1-sub:x2+sub])
            cropped_img = image[y1-sub:y2+sub, x1-sub:x2+sub]
            cv2.imwrite(save_path, cropped_img)
            
            # 姿态估计
            result = inference_topdown(model_pose, save_path)
            pose_sample = result[0]
            
            # 从 pred_instances 中取出 keypoints
            if hasattr(pose_sample.pred_instances, 'keypoints'):
                keypoints = pose_sample.pred_instances.keypoints  # shape: (1, N, 2)
                keypoints_np = keypoints[0]  # shape: (N, 2)
                
                if has_duplicate_keypoints(keypoints_np):
                    print(keypoints)
                    # boxes, result, gray_image = process_gray_image(output_image)
                    # if boxes is None or result is None or gray_image is None or len(boxes) != 2:
                        # cv2.rectangle(result, box[0], box[2], (0, 255, 0), 2)
                        # for box in boxes:
                    subpix_corners = get_subpixel_corners_near_vertices(image,box)
                    for (x, y) in subpix_corners:
                        cv2.circle(output_image, (int(round(x)), int(round(y))), 1, (0, 0, 255), -1)
                #print(f"关键点数量: {len(keypoints_np)}")
                
                
                # 在裁剪图像上绘制关键点
                for (x, y) in keypoints_np:
                    # 只绘制有效的关键点（排除负坐标）
                    x = int(x)
                    y = int(y)
                    x = max(x, 0)
                    y = max(y, 0)
                    x = min(x2 - x1 + 2*sub - 1, x)
                    y = min(y2 - y1 + 2*sub - 1, y)
                    cv2.circle(cropped_img, (x, y), 1, (0, 0, 255), -1)
                
                # 在原始图像上绘制关键点（需要转换坐标）
                for (x, y) in keypoints_np:
                    x = int(x) + (x1 - sub)  # 转换到原始图像坐标
                    y = int(y) + (y1 - sub)  # 转换到原始图像坐标
                    # 确保坐标在原始图像范围内
                    x = max(x, 0)
                    y = max(y, 0)
                    x = min(img_width - 1, x)
                    y = min(img_height - 1, y)
                    print(output_image.shape,x,y)
                    cv2.circle(output_image, (x, y), 1, (0, 255, 0), -1)  # 使用不同颜色区分

            # 保存带有关键点的裁剪图像
            cv2.imwrite(save_path, cropped_img)
            print(i, save_path)

        # 保存带有所有关键点的原始图像
        full_image_save_path = os.path.join(output_folder2, f"{i}_full_image_with_keypoints.jpg")
        cv2.imwrite(full_image_save_path, output_image)
        print(f"已保存带有关键点的完整图像: {full_image_save_path}")