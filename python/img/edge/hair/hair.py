import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def process_hair_image(image_path, save_plot_dir="hair_analysis_plots"):
    """
    Process a single hair image: extract flat-segment thickness + save visualization
    :param image_path: Path to the image
    :param save_plot_dir: Directory to save visualization results
    :return: Hair count, average thickness, min thickness, max thickness
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_plot_dir, exist_ok=True)

    # 1. Read image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Failed to read image: {image_path}")
        return None, None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Adaptive thresholding (optimized to reduce false detection)
    mask = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,  # Larger block size to reduce small noise
        C=8            # Larger offset to extract darker hairs strictly
    )
    # Morphological opening (filter small noise)
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 3. Canny edge detection
    edges = cv2.Canny(mask_clean, 30, 100)
    # 4. Connected component analysis (count hairs + calculate flat-segment thickness)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_clean, connectivity=8
    )
    hair_count = num_labels - 1  # Exclude background (label=0)

    valid_hair_labels = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]  # 获取当前连通域的面积
        if area >= 40:  # 只保留面积≥阈值的连通域
            valid_hair_labels.append(label)
    hair_count = len(valid_hair_labels)  # 有效毛发数（替换原来的num_labels-1）

    hair_thickness = []
    for label in valid_hair_labels:  # 仅遍历有效毛发的label
        # Extract single hair mask
        single_hair = (labels == label).astype(np.uint8) * 255
        # Get all pixel coordinates of the hair
        y_coords, x_coords = np.where(single_hair == 255)
        if len(x_coords) < 10:  # 额外过滤过短的毛发（可选）
            continue

        # Calculate hair direction (PCA to find flat segment)
        coords = np.column_stack((x_coords, y_coords))
        mean_coords = np.mean(coords, axis=0)
        centered_coords = coords - mean_coords
        cov_matrix = np.cov(centered_coords.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Main direction (hair extension direction)
        main_dir = eigenvectors[np.argmax(eigenvalues)]

        # Project to axis perpendicular to main direction, calculate thickness (flat segment)
        perp_dir = np.array([-main_dir[1], main_dir[0]])
        perp_dir = perp_dir / np.linalg.norm(perp_dir)  # Normalize
        # Projection coordinates on perpendicular axis
        projections = np.dot(centered_coords, perp_dir)
        # Flat segment thickness = max - min (exclude 10% at both ends)
        projections_sorted = np.sort(projections)
        len_proj = len(projections_sorted)
        start_idx = int(len_proj * 0.1)
        end_idx = int(len_proj * 0.9)
        if end_idx - start_idx < 3:
            continue
        thick = projections_sorted[end_idx] - projections_sorted[start_idx]
        hair_thickness.append(thick)

    # Calculate statistics
    avg_thickness = np.mean(hair_thickness) if hair_thickness else 0
    min_thickness = np.min(hair_thickness) if hair_thickness else 0
    max_thickness = np.max(hair_thickness) if hair_thickness else 0

    # Save visualization to local directory
    plt.figure(figsize=(12, 6))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Original Image")
    plt.subplot(132), plt.imshow(mask_clean, cmap="gray"), plt.title("Hair Mask")
    plt.subplot(133), plt.imshow(edges, cmap="gray"), plt.title("Canny Edges")
    plt.suptitle(f"Image: {os.path.basename(image_path)} | Hair Count: {hair_count}", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_plot_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_analysis.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()  # Close figure to free memory

    # Print single image result
    print(f"\n=== Image: {os.path.basename(image_path)} ===")
    print(f"Detected hair count: {hair_count}")
    if hair_thickness:
        print(f"Average hair thickness (flat segment, pixels): {avg_thickness:.2f}")
        print(f"Hair thickness range (flat segment, pixels): {min_thickness:.2f} ~ {max_thickness:.2f}")
    else:
        print("No valid hair thickness data detected")

    return hair_count, avg_thickness, min_thickness, max_thickness

def batch_process_hair_images(folder_path="data", save_plot_dir="hair_analysis_plots"):
    """Batch process images and save visualization results"""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    if not image_paths:
        print(f"❌ No image files found in {folder_path}")
        return

    total_images = len(image_paths)
    total_hair_count = 0
    all_avg_thickness = []
    all_min_thickness = []
    all_max_thickness = []

    print(f"\n📁 Starting batch processing for images in {folder_path}...")
    print(f"Found {total_images} images")
    print("-" * 50)

    for img_path in image_paths:
        count, avg, min_t, max_t = process_hair_image(img_path, save_plot_dir)
        if count is not None:
            total_hair_count += count
            if avg > 0:
                all_avg_thickness.append(avg)
                all_min_thickness.append(min_t)
                all_max_thickness.append(max_t)

    # Print summary result
    print("\n" + "=" * 60)
    print("📊 Batch Processing Summary")
    print("=" * 60)
    print(f"Total images processed: {total_images}")
    print(f"Total hair count across all images: {total_hair_count}")
    if all_avg_thickness:
        overall_avg = np.mean(all_avg_thickness)
        overall_min = np.min(all_min_thickness)
        overall_max = np.max(all_max_thickness)
        print(f"Overall average hair thickness (flat segment, pixels): {overall_avg:.2f}")
        print(f"Overall hair thickness range (flat segment, pixels): {overall_min:.2f} ~ {overall_max:.2f}")
    else:
        print("No valid hair thickness data available")
    print(f"Visualization results saved to: {os.path.abspath(save_plot_dir)}")
    print("=" * 60)

if __name__ == "__main__":
    # Batch process images in "data" folder, save visualizations to "hair_analysis_plots"
    batch_process_hair_images(folder_path="data", save_plot_dir="hair_analysis_plots")