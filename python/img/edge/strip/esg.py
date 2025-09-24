import cv2
import numpy as np
import os


def normalize_coords(x, y, img_w, img_h):
    """将坐标归一化到0~1范围（相对于图像宽高）"""
    norm_x = x / img_w if img_w != 0 else 0.0
    norm_y = y / img_h if img_h != 0 else 0.0
    return np.clip(norm_x, 0.0, 1.0), np.clip(norm_y, 0.0, 1.0)


import cv2
import numpy as np

import cv2
import numpy as np
from math import atan2, degrees, radians

def get_subpixel_corners_near_vertices(image, box, search_radius=5):
    """结合外接框（box）信息，在box内部的扇形区域检测亚像素角点"""
    # 1. 灰度转换与对比度增强（突出白色角点特征）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = clahe.apply(gray_img)

    # 2. 基于灰度值的白色区域筛选（适应绿色关键点周围的亮度特征）
    _, white_mask = cv2.threshold(
        gray_img, 
        thresh=100,  # 保留绿色关键点附近的中高亮度区域（可微调）
        maxval=255, 
        type=cv2.THRESH_BINARY
    )

    # 3. 掩码优化：保留与box边缘相关的白色区域
    kernel = np.ones((2, 2), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)  # 去噪
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)  # 填充小空洞

    # 预处理：计算box的边界范围（确保角点在box内部）
    box_np = np.array(box, dtype=np.int32)
    min_x, min_y = np.min(box_np[:, 0]), np.min(box_np[:, 1])
    max_x, max_y = np.max(box_np[:, 0]), np.max(box_np[:, 1])

    subpix_corners = []
    for i, (x, y) in enumerate(box):
        # 4. 确定box顶点的相邻边方向（用于生成扇形搜索区域）
        prev_idx = (i - 1) % len(box)
        next_idx = (i + 1) % len(box)
        prev_x, prev_y = box[prev_idx]
        next_x, next_y = box[next_idx]

        # 计算两条相邻边的方向向量（用于约束扇形角度）
        edge_prev = (prev_x - x, prev_y - y)  # 前一条边向量
        edge_next = (next_x - x, next_y - y)  # 后一条边向量

        # 5. 定义局部搜索区域（box内部，半径search_radius）
        x_start = max(min_x, x - search_radius)  # 限制在box左边界内
        x_end = min(max_x, x + search_radius + 1)  # 限制在box右边界内
        y_start = max(min_y, y - search_radius)  # 限制在box下边界内
        y_end = min(max_y, y + search_radius + 1)  # 限制在box上边界内

        # 提取局部ROI
        roi_gray = gray_img[y_start:y_end, x_start:x_end]
        roi_white = white_mask[y_start:y_end, x_start:x_end]
        if roi_gray.size == 0 or roi_white.size == 0:
            subpix_corners.append(np.array([x, y], dtype=np.float32))
            continue

        # 6. 生成扇形掩码（沿box边方向开口，仅在两条边之间搜索）
        roi_h, roi_w = roi_gray.shape
        fan_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        center_x, center_y = x - x_start, y - y_start  # 扇形中心在ROI中的坐标

        for dy in range(roi_h):
            for dx in range(roi_w):
                # 计算像素相对中心的向量
                px = dx - center_x
                py = dy - center_y
                if px == 0 and py == 0:
                    fan_mask[dy, dx] = 1  # 中心像素有效
                    continue

                # 判断像素是否在两条边形成的夹角内（扇形区域）
                # 点积判断方向：在edge_prev和edge_next之间
                dot_prev = px * edge_prev[0] + py * edge_prev[1]
                dot_next = px * edge_next[0] + py * edge_next[1]
                if dot_prev >= 0 and dot_next >= 0:  # 在两边夹角内
                    fan_mask[dy, dx] = 1

        # 7. 合并掩码：仅保留box内、扇形区域内的白色区域
        combined_roi = roi_gray.copy()
        combined_roi[np.logical_or(roi_white == 0, fan_mask == 0)] = 0

        # 8. 检测角点（优先选择符合box边缘特征的角点）
        corners = cv2.goodFeaturesToTrack(
            combined_roi,
            maxCorners=3,  # 允许多个候选，提高检出率
            qualityLevel=0.005,  # 降低阈值，适应绿色关键点附近的弱纹理
            minDistance=1,
            blockSize=3
        )

        if corners is not None and len(corners) > 0:
            # 转换坐标到原图
            corners = np.int32(corners).reshape(-1, 2)
            corners = [(cx + x_start, cy + y_start) for (cx, cy) in corners]

            # 筛选：角点必须在box内部
            valid_corners = []
            for (cx, cy) in corners:
                if min_x <= cx <= max_x and min_y <= cy <= max_y:
                    valid_corners.append((cx, cy))

            if valid_corners:
                # 选择离原始顶点最近且符合box边方向的角点
                corners_with_dist = [
                    ((cx - x)**2 + (cy - y)** 2, cx, cy) 
                    for (cx, cy) in valid_corners
                ]
                corners_with_dist.sort()
                nearest_corner = (corners_with_dist[0][1], corners_with_dist[0][2])

                # 亚像素细化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                subpix = cv2.cornerSubPix(
                    gray_img,
                    np.float32([nearest_corner]),
                    winSize=(3, 3),
                    zeroZone=(-1, -1),
                    criteria=criteria
                )
                subpix_corners.append(subpix[0])
                continue

        # 9. 未检测到角点时，使用box边缘方向的插值点
        # 沿两条边的方向各取一点，选择更近的作为候选
        edge_prev_point = (x + edge_prev[0]//2, y + edge_prev[1]//2)  # 前边上的点
        edge_next_point = (x + edge_next[0]//2, y + edge_next[1]//2)  # 后边上的点

        # 确保候选点在box内部
        candidates = []
        if min_x <= edge_prev_point[0] <= max_x and min_y <= edge_prev_point[1] <= max_y:
            candidates.append(edge_prev_point)
        if min_x <= edge_next_point[0] <= max_x and min_y <= edge_next_point[1] <= max_y:
            candidates.append(edge_next_point)

        if candidates:
            # 选择离原始顶点最近的候选点
            cand_with_dist = [((cx - x)**2 + (cy - y)** 2, cx, cy) for (cx, cy) in candidates]
            cand_with_dist.sort()
            nearest_cand = (cand_with_dist[0][1], cand_with_dist[0][2])
            subpix_corners.append(np.array(nearest_cand, dtype=np.float32))
            continue

        # 10. 最终fallback：使用原始顶点
        subpix_corners.append(np.array([x, y], dtype=np.float32))

    return np.array(subpix_corners, dtype=np.float32)






def process_gray_image(image):
    # image = cv2.imread(image_path)
    # if image is None:
    #     print(f"图片 {image_path} 未找到或不是有效图像！")
    #     return None, None, None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    result = image.copy()  # 用于绘制结果的彩色图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    valid_boxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        rotated_rect = cv2.minAreaRect(contour)
        (_, (width, height), _) = rotated_rect
        # 过滤过小/过大的轮廓
        if area < 50 or min(width, height) < 5 or area > 450:
            continue
        
        box = cv2.boxPoints(rotated_rect)
        box = np.int32(box)
        x_min, y_min = np.min(box, axis=0)
        x_max, y_max = np.max(box, axis=0)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1] - 1, x_max)
        y_max = min(image.shape[0] - 1, y_max)

        aspect_ratio = max(width, height) / min(width, height)
        roi_gray = gray[y_min:y_max, x_min:x_max]
        mean_brightness = np.mean(roi_gray) if roi_gray.size > 0 else 0
        roi_white = white_mask[y_min:y_max, x_min:x_max]
        white_area = cv2.countNonZero(roi_white)
        total_area = (x_max - x_min) * (y_max - y_min)
        if total_area == 0:
            continue
        ratio = white_area / total_area
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        num = len(approx)

        # 筛选矩形或高亮度目标，且宽高比符合条件
        if (num == 4 or mean_brightness > 70) and 2 < aspect_ratio < 5:
            valid_boxes.append((aspect_ratio, box))

    # 确保输出2个框
    if len(valid_boxes) == 2:
        for _, box in valid_boxes:
            boxes.append(box)
    else:
        valid_boxes.sort(key=lambda x: x[0], reverse=True)
        top_one_boxes = [box for _, box in valid_boxes[:1]]
        extra_boxes = [box for _, box in valid_boxes[1:]]

        for box in top_one_boxes:
            boxes.append(box)

        if extra_boxes:
            all_points = np.concatenate(extra_boxes, axis=0)
            merged_rect = cv2.minAreaRect(all_points)
            merged_box = cv2.boxPoints(merged_rect)
            merged_box = np.int32(merged_box)
            boxes.append(merged_box)
    
    return boxes, result, gray


def traverse_gray_folder(input_folder, output_folder, label_folder):
    # 创建输出和标签文件夹
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            save_path = os.path.join(output_folder, filename)
            # 标签文件路径（与图像同名，替换后缀为txt）
            label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')
            
            # 获取检测到的矩形框和图像
            boxes, result, gray_image = process_gray_image(image_path)
            if boxes is None or result is None or gray_image is None or len(boxes) != 2:
                # 若不符合条件，生成空标签文件
                with open(label_path, 'w') as f:
                    pass
                continue
            
            # 获取图像宽高（用于归一化）
            img_h, img_w = gray_image.shape[:2]
            if img_w == 0 or img_h == 0:
                continue
            
            # 打开标签文件准备写入
            with open(label_path, 'w') as f:
                # 处理每个矩形框（每个框对应一个实例）
                for box in boxes:
                    # 1. 计算边界框的归一化参数（基于最小外接矩形）
                    box_int = np.int32(box)
                    # 计算最小外接接正矩形（x,y为左上角坐标，w,h为宽高）
                    x, y, w, h = cv2.boundingRect(box_int)
                    
                    # 2. 计算正矩形的中心点心坐标
                    cx = x + w / 2  # 中心x坐标 = 左上角x + 宽/2
                    cy = y + h / 2  # 中心y坐标 = 左上角y + 高/2
                    
                    # 3. 归一化参数（转换为0~1范围）
                    norm_cx, norm_cy = normalize_coords(cx, cy, img_w, img_h)
                    norm_width = w / img_w if img_w != 0 else 0.0
                    norm_height = h / img_h if img_h != 0 else 0.0
                    norm_width = np.clip(norm_width, 0.0, 1.0)
                    norm_height = np.clip(norm_height, 0.0, 1.0)
                    
                    # 2. 获取亚像素角点并处理关键点
                    subpix_corners = get_subpixel_corners_near_vertices(gray_image, box)
                    kp_data = []
                    for (x_kp, y_kp) in subpix_corners:
                        # 归一化关键点坐标
                        norm_kp_x, norm_kp_y = normalize_coords(x_kp, y_kp, img_w, img_h)
                        # 关键点可见性：亚像素角点清晰可见，设为2
                        kp_visibility = 2  
                        kp_data.extend([norm_kp_x, norm_kp_y, kp_visibility])
                    
                    # 3. 构建YOLOv11 Pose标签行
                    class_id = 0  # 固定类别为transistor（0）
                    conf = 0.3    # 目标置信度（可根据实际调整）
                    # 拼接标签内容（保留5位小数，与示例格式一致）
                    label_line = [
                        str(class_id),
                        f"{norm_cx:.5f}", f"{norm_cy:.5f}",
                        f"{norm_width:.5f}", f"{norm_height:.5f}"
                        # f"{conf:.5f}"
                    ]
                    # 添加关键点数据（坐标+可见性，可见性转为整数）
                    label_line.extend([
                        f"{kp:.5f}" if i % 3 != 2 else str(int(kp)) 
                        for i, kp in enumerate(kp_data)
                    ])
                    # 写入标签文件（每行一个实例）
                    f.write(' '.join(label_line) + '\n')
            
            # 绘制亚像素角点并保存可视化结果
            for box in boxes:
                subpix_corners = get_subpixel_corners_near_vertices(gray_image, box)
                for (x, y) in subpix_corners:
                    cv2.circle(result, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
            cv2.imwrite(save_path, result)
            print(f"已处理：{filename}，结果保存至：{save_path}，标签保存至：{label_path}")


if __name__ == "__main__":
    input_folder = "/home/dreame/dmt/data/nvme7/dataset/pose/strip/data/"  # 输入图像文件夹
    output_folder = "/home/dreame/dmt/program/project/apriltagone-MOVA3000_two_rect/result"  # 可视化结果文件夹
    label_folder = "/home/dreame/dmt/data/nvme7/dataset/pose/strip/labels"  # 标签保存文件夹
    traverse_gray_folder(input_folder, output_folder, label_folder)