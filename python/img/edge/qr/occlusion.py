import cv2
import numpy as np
import os
from vision import ObjectDetector

background = None
def build_background(image, alpha=0.1):
    """建立平滑背景图像"""
    global background
    if background is None:
        background = np.float32(image)
    else:
        cv2.accumulateWeighted(image, background, alpha)

def extract_high_frequency_component(image, background):
    """提取高频分量"""
    diff = cv2.absdiff(image, cv2.convertScaleAbs(background))
    return diff

def build_general_image(image, background, enhance=True):
    """建立广义图像"""
    if enhance:
        image = cv2.equalizeHist(image)
    background = cv2.cvtColor(cv2.convertScaleAbs(background), cv2.COLOR_BGR2GRAY)
    general_image = cv2.absdiff(image, background)
    return general_image

def edge_detection(image):
    """边缘提取"""
    return cv2.Canny(image, 100, 200)

def detect_suspected_occlusion(edge_image, threshold=30):
    """疑似遮挡区域检测"""
    _, binary = cv2.threshold(edge_image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def determine_occlusion(binary_image, frame, threshold=10):
    """遮挡区域确定"""
    occluded_regions = []
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > threshold:
            occluded_regions.append(contour)
    return occluded_regions

# 主处理流程
def process_frame(frame):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    build_background(frame)  # 更新背景
    high_freq = extract_high_frequency_component(frame, background)
    general_image = build_general_image(gray_frame, background)
    edge_img = edge_detection(general_image)
    suspected_occlusions = detect_suspected_occlusion(edge_img)
    occlusions = determine_occlusion(suspected_occlusions, frame)
    
    return occlusions

def calculate_iou(box1, box2):
    """计算两个矩形框的IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (w1 * h1)
    boxBArea = (w2 * h2)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_center(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def merge_nearby_boxes(boxes, threshold):
    merged_boxes = []
    boxes = list(boxes)
    
    while boxes:
        box = boxes.pop(0)
        center = calculate_center(box)
        merge_list = [box]
        
        for other_box in boxes:
            other_center = calculate_center(other_box)
            if distance(center, other_center) < threshold:
                merge_list.append(other_box)
        
        boxes = [b for b in boxes if b not in merge_list]
        x_min = min([b[0] for b in merge_list])
        y_min = min([b[1] for b in merge_list])
        x_max = max([b[0] + b[2] for b in merge_list])
        y_max = max([b[1] + b[3] for b in merge_list])
        
        merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
    
    return merged_boxes

def process_frame_with_merging(frame, occlusions, bboxs,threshold=50):
    # 将遮挡区域合并
    boxes = [cv2.boundingRect(occlusion) for occlusion in occlusions if cv2.contourArea(occlusion) >= 100]
    
    merged_boxes = merge_nearby_boxes(boxes, threshold)
    
      # 将合并后的矩形框与检测框进行IoU比较
    updated_boxes = []
    for (x, y, w, h) in merged_boxes:
        is_merged = False
        for (bx, by, bw, bh) in bboxs:
            iou = calculate_iou((x, y, w, h), (bx, by, bw, bh))
            if iou > 0.2:
                x = min(x, bx)
                y = min(y, by)
                w = max(x + w, bx + bw) - x
                h = max(y + h, by + bh) - y
                is_merged = True
                break
        if is_merged:
            updated_boxes.append((x, y, w, h))
    
    # 保留原有检测框
    # updated_boxes.extend(bboxs)
    
    # 在图像上绘制所有矩形框
    for (x, y, w, h) in updated_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

import argparse
parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network of MNN ')
parser.add_argument("--video", help="path to video file. ")
parser.add_argument("--img",help='source img path ' )
parser.add_argument("--img_dir",help='source img path ' )
parser.add_argument("--save_dir", default="./test/save_mnn/",help='save img path ')
parser.add_argument("--model", default="/home/morgan/ubt/alg/deploy/MNN/build-x86/model/mobilnetssd_wqint8.mnn",help='Path to model')
parser.add_argument("--lite_model", default="/home/morgan/ubt/program/project/ai-4-qcm2290-no-snpe/file-copy/data/ai/grass/tflite_mssdint8_grass.mnn",help='Path to model')
parser.add_argument("--prototxt", default="/home/morgan/ubt/alg/cv/export/MobileNet-SSD/example/MobileNetSSD_deploy.prototxt",help='Path to prototxt')
parser.add_argument("--weights", default="/home/morgan/ubt/alg/cv/export/MobileNet-SSD/snapshot_5/mobilenet_iter_110000.caffemodel",help='Path to weights')
parser.add_argument("--thr", default=0.35, type=float, help="confidence threshold to filter out weak detections")


args = parser.parse_args()
detector = ObjectDetector(args.model,args.lite_model,args.prototxt , args.weights,thr=args.thr)

image_path = "/home/morgan/ubt/data/0722/0803测试/4"
save_path = "/home/morgan/ubt/data/0722/0803测试/test"
for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(root, file)
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            bboxs = detector.infer_lite(frame)
            occlusions = process_frame(frame)
            for bbox in bboxs:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for occlusion in occlusions:
                box = cv2.boundingRect(occlusion)
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 255, 0), 2)
            frame = process_frame_with_merging(frame, occlusions,bboxs)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imwrite(os.path.join(save_path,file), frame)
cv2.destroyAllWindows()