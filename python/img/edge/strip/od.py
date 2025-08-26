import cv2
import numpy as np
from ultralytics import YOLO
import os
from esg import get_subpixel_corners_near_vertices, process_gray_image
model = YOLO("/home/dreame/dmt/data/nvme7/alg/cv/detection/ultralytics/runs/train/exp16/weights/best.pt")  
input_folder = "/home/dreame/dmt/program/project/apriltagone-MOVA3000_two_rect/data"  # 输入图像文件夹
output_folder = "/home/dreame/dmt/data/nvme7/dataset/pose/strip/result0"  # 可视化结果文件夹
os.makedirs(output_folder, exist_ok=True)

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


i = 0
sub = 3
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
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
            # boxes, result, gray_image = process_gray_image(image[y1-5:y2+5, x1-5:x2+5])
            # if boxes is None or result is None or gray_image is None or len(boxes) != 2:
            #     # cv2.rectangle(result, box[0], box[2], (0, 255, 0), 2)
            #     for box in boxes:
            #         subpix_corners = get_subpixel_corners_near_vertices(gray_image,box)
            #         # for (x, y) in subpix_corners:
            #         #     cv2.circle(result, (int(round(x)), int(round(y))), 1, (0, 0, 255), -1)
            i += 1

            save_path = os.path.join(output_folder, f"{i}_strip_0818.jpg")
            cv2.imwrite(save_path,image[y1-sub:y2+sub, x1-sub:x2+sub])
