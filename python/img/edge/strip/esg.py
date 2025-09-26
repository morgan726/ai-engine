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

import cv2
import numpy as np

def get_subpixel_corners_near_vertices(image, box, keypoints_np, search_radius=5, dist_threshold=3.0):
    """
    基于box顶点向内扇区补全角点：
    1. 检查每个box顶点的向内四分之一扇区，若存在姿态估计的关键点（keypoints_np），直接复用
    2. 若无关键点，检测白色区域的轮廓顶点；无白色区域则用原始顶点
    参数：
        image: 原图（BGR格式）
        box: 目标外接框顶点列表（4个点，(x,y)，原图坐标，int）
        keypoints_np: 姿态估计得到的角点（N,2，原图坐标，np.array）
        search_radius: 扇区半径（默认5像素）
        dist_threshold: 关键点与扇区中心的距离阈值（默认3像素，判断是否在扇区内）
    返回：
        subpix_corners: 补全后的亚像素角点（4,2，np.float32，原图坐标）
    """
    # 1. 灰度预处理（增强白色区域对比度）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = clahe.apply(gray_img)

    # 2. 提取白色区域（适配白色物体特征）
    _, white_mask = cv2.threshold(
        gray_img, 
        thresh=50,  # 白色物体阈值（可根据实际白色深浅微调）
        maxval=255, 
        type=cv2.THRESH_BINARY
    )

    # 3. 优化白色掩码（去噪+填充，避免小空洞影响轮廓检测）
    kernel = np.ones((2, 2), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)  # 去小噪点
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)  # 填小空洞

    # 4. 计算box整体边界（确保所有点在box内部）
    box_np = np.array(box, dtype=np.int32)
    min_x, min_y = np.min(box_np[:, 0]), np.min(box_np[:, 1])
    max_x, max_y = np.max(box_np[:, 0]), np.max(box_np[:, 1])

    subpix_corners = []
    for i, (x_center, y_center) in enumerate(box):
        # -------------------------- 步骤1：定义当前顶点的向内四分之一扇区 --------------------------
        # 计算指向box内部的相邻边向量（核心：确保扇区朝向box内）
        prev_idx = (i - 1) % 4  # box是4个顶点，循环取相邻点
        next_idx = (i + 1) % 4
        prev_x, prev_y = box[prev_idx]
        next_x, next_y = box[next_idx]
        edge_prev = (x_center - prev_x, y_center - prev_y)  # 从相邻点指向当前顶点（向内）
        edge_next = (x_center - next_x, y_center - next_y)  # 从相邻点指向当前顶点（向内）

        # 定义扇区的局部搜索范围（限制在box内部，避免超出目标区域）
        x_start = max(min_x, x_center - search_radius)
        x_end = min(max_x, x_center + search_radius + 1)
        y_start = max(min_y, y_center - search_radius)
        y_end = min(max_y, y_center + search_radius + 1)

        # -------------------------- 步骤2：检查扇区内是否存在姿态估计的关键点 --------------------------
        found_keypoint = None
        if len(keypoints_np) > 0:
            for (kp_x, kp_y) in keypoints_np:
                # 1. 先判断关键点是否在扇区的矩形搜索范围内
                if not (x_start <= kp_x <= x_end and y_start <= kp_y <= y_end):
                    continue
                # 2. 计算关键点相对扇区中心的向量（判断是否在向内的夹角内）
                vec_kp = (kp_x - x_center, kp_y - y_center)
                dot_prev = vec_kp[0] * edge_prev[0] + vec_kp[1] * edge_prev[1]  # 与前向边的点积（>0表示同向向内）
                dot_next = vec_kp[0] * edge_next[0] + vec_kp[1] * edge_next[1]  # 与后向边的点积（>0表示同向向内）
                # 3. 计算关键点与中心的距离（确保在扇区半径内）
                dist_sq = (kp_x - x_center)**2 + (kp_y - y_center)** 2
                if dot_prev >= 0 and dot_next >= 0 and dist_sq <= (search_radius + dist_threshold)**2:
                    found_keypoint = (kp_x, kp_y)
                    break  # 找到一个有效关键点即可，无需继续遍历

        # -------------------------- 步骤3：根据是否找到关键点，确定最终角点 --------------------------
        if found_keypoint is not None:
            # 情况1：扇区内有关键点，直接对该点做亚像素细化（提升精度）
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            subpix = cv2.cornerSubPix(
                gray_img,
                np.float32([found_keypoint]),
                winSize=(3, 3),
                zeroZone=(-1, -1),
                criteria=criteria
            )
            # subpix_corners.append(subpix[0])
            continue

        # 情况2：扇区内无关键点，从白色区域提取轮廓顶点
        # 提取扇区局部的白色掩码ROI
        roi_white = white_mask[y_start:y_end, x_start:x_end]
        if roi_white.size == 0:
            # 无白色区域，fallback到原始顶点
            subpix_corners.append(np.array([x_center, y_center], dtype=np.float32))
            continue

        roi_gray = gray_img[y_start:y_end, x_start:x_end]
        # 角点检测参数：只检测1个最可能的角点，提高质量阈值
        corners = cv2.goodFeaturesToTrack(
            roi_gray,
            maxCorners=1,
            qualityLevel=0.01,
            minDistance=5,
            mask=roi_white  # 只在白色区域内检测
        )

        best_vertex = None
        if corners is not None and len(corners) > 0:
            # 将ROI内的角点坐标转换为原图坐标
            corner = corners[0][0] + np.array([x_start, y_start])
            cx, cy = corner
            # 检查角点是否在box内部
            if min_x <= cx <= max_x and min_y <= cy <= max_y:
                best_vertex = (cx, cy)

        # 如果角点检测失败或角点位置不合适，再尝试轮廓方法
        if best_vertex is None:
            # 检测白色区域的轮廓（优先取面积最大的轮廓，避免小噪点干扰）
            contours, _ = cv2.findContours(roi_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                # 无轮廓，fallback到原始顶点
                subpix_corners.append(np.array([x_center, y_center], dtype=np.float32))
                continue

            # 取面积最大的轮廓，拟合其最小外接矩形（获取轮廓顶点）
            largest_contour = max(contours, key=cv2.contourArea)
            # 转换轮廓坐标到原图（roi坐标 -> 原图坐标）
            largest_contour = largest_contour + np.array([[x_start, y_start]])  # 轮廓点偏移
            rect = cv2.minAreaRect(largest_contour)  # 拟合最小外接矩形
            contour_vertices = cv2.boxPoints(rect)  # 获取矩形4个顶点
            contour_vertices = np.int32(contour_vertices)  # 转为int坐标

            # 选择轮廓顶点中离当前box顶点最近的点（作为补全角点）
            min_dist = float('inf')
            best_vertex = (x_center, y_center)  # 默认值
            for (vx, vy) in contour_vertices:
                # 确保轮廓顶点在box内部（避免取到外部干扰点）
                if not (min_x <= vx <= max_x and min_y <= vy <= max_y):
                    continue
                dist = (vx - x_center)**2 + (vy - y_center)** 2
                if dist < min_dist:
                    min_dist = dist
                    best_vertex = (vx, vy)

        # 对找到的顶点做亚像素细化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        subpix = cv2.cornerSubPix(
            gray_img,
            np.float32([best_vertex]),
            winSize=(3, 3),
            zeroZone=(-1, -1),
            criteria=criteria
        )
        subpix_corners.append(subpix[0])

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