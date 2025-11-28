import cv2
import numpy as np
import sys

save_path = "/home/morgan/ubt/data/log/05/0514/1/"

def merge_contours(edges, morph_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
count = 0

def main(image_path):
    global count
    count += 1
    print("--------------------------------------",count)
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 100, 150)
    merged_contours = merge_contours(edges)
    num = 0
    is_qr = False
    for contour in merged_contours:
        area = cv2.contourArea(contour)
        print(area)
        if area < 1000 or area > 30000:
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            distances = []
            points = approx.reshape(4, 2)
            for i in range(4):
                p1 = points[i]
                p2 = points[(i+1) % 4]  # 下一个点，形成闭环
                dist = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
                distances.append(dist)
            max_side = max(distances)
            min_side = min(distances)
            aspect_ratio = max_side / min_side
            print(area,aspect_ratio)
            if aspect_ratio < 2 :
                center_x = int(np.mean([p[0] for p in points]))
                center_y = int(np.mean([p[1] for p in points]))
                print(f"轮廓中心点: ({center_x}, {center_y})")
                print("相机在充电桩的正前方")
                num += 1
                is_qr =True
                cv2.drawContours(result, [approx], -1, (0, 255, 5), 5)
    if is_qr:
        print(count, ":",num)
    if not is_qr:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        kernel = np.ones((5, 5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
        allrect = cv2.minAreaRect(all_points)
        (acx, acy), (width, height), all_angle = allrect
        all_aspect_ratio = max(width,height) / min(width,height)
        box = cv2.boxPoints(allrect)
        box = np.int0(box)

        result_image = image.copy()
        cv2.drawContours(result_image, [box], 0, (0, 0, 255), 2)
        cv2.drawContours(result_image, contours, -1, (0, 255, 5), 5)
        max_height = 0
        tallest_contour = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > max_height:
                max_height = h
                tallest_contour = contour
        print(black_mask.shape)
        if tallest_contour is not None:
            aspect_ratio = black_mask.shape[1] / black_mask.shape[0]
            print(aspect_ratio)
            rect = cv2.minAreaRect(tallest_contour)
            (cx, cy), (width, height), angle = rect
            img_width = black_mask.shape[1]

            if width / img_width > 0.6:
                if cx < img_width / 2:
                    print("相机在充电桩的正右侧")
                else:
                    print("相机在充电桩的正左侧")
            else:
                if aspect_ratio > 2:
                    if cx < 1 * img_width / 3:
                        print("相机在充电桩的左后侧")
                    elif cx > 2* img_width / 3:
                        print("相机在充电桩的右后侧")
                    else:
                        print("相机在充电桩的后正中")
                else:
                    if cx < 1 * img_width / 3:
                        print("相机在充电桩的左前侧")
                    elif cx > 2* img_width / 3:
                        print("相机在充电桩的右前侧")
                    else:
                        print("相机在充电桩的前正中")
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(result_image, [box], 0, (0, 255, 5), 5)  
            cv2.imwrite(save_path + str(count) + '_result_image.jpg', result_image)
            
            print(f"中心点位置: {cx}")
        else:
            print("未能确定底座轮廓")
    cv2.imwrite(save_path + str(count) + "_merged_contours_with_min_rects.jpg", result)

if __name__ == "__main__":
    import os
    for root, dirs, files in os.walk(sys.argv[1]):
        for file in files:
            if file.endswith(".jpg"):
                print(file)
                image_path = os.path.join(root, file)
                main(image_path)