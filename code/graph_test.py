import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_flocs_by_sharpness(image_path, crop_ratio=0.7, min_floc_area=5, max_floc_area=300, edge_thresh=25):
    """
    使用边缘梯度（清晰度）方法分析矾花图像：过滤高亮但模糊的焦外背景。
    
    :param image_path: 图片路径
    :param crop_ratio: 中心截取比例
    :param min_floc_area: 最小有效面积阈值
    :param max_floc_area: 最大有效面积阈值
    :param edge_thresh: 边缘锐度阈值。越大越挑剔，只保留最清晰的矾花。
    """
    # 1. 读取与截取
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"错误：无法读取图片 {image_path}")
        return None

    height, width = original_img.shape[:2]
    crop_h, crop_w = int(height * crop_ratio), int(width * crop_ratio)
    y1, x1 = int((height - crop_h) / 2), int((width - crop_w) / 2)
    cropped_img = original_img[y1:y1+crop_h, x1:x1+crop_w]

    # 2. 灰度化与轻微降噪
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 3. 提取拉普拉斯梯度（寻找锐利边缘）
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    sharpness_map = cv2.convertScaleAbs(laplacian)
    
    # 4. 二值化梯度图：只保留真正锐利的像素
    _, edge_mask = cv2.threshold(sharpness_map, edge_thresh, 255, cv2.THRESH_BINARY)
    
    # 5. 形态学闭操作与轮廓填充
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(gray)
    cv2.drawContours(solid_mask, contours, -1, 255, thickness=cv2.FILLED)

    # 6. 特征统计
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(solid_mask, connectivity=8)

    valid_floc_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_floc_area and area < max_floc_area:
            valid_floc_areas.append(area)

    floc_count = len(valid_floc_areas)
    cropped_area_px = crop_w * crop_h
    floc_density = floc_count / cropped_area_px if cropped_area_px > 0 else 0

    # 7. 控制台格式化输出
    print(f"\n--- 基于清晰度的矾花分析结果 ---")
    if floc_count > 0:
        max_floc_area_val = max(valid_floc_areas)
        min_floc_area_val = min(valid_floc_areas)
    else:
        max_floc_area_val = 0
        min_floc_area_val = 0

    print(f"有效矾花数量: {floc_count}")
    print(f"最大矾花面积: {max_floc_area_val} 像素")
    print(f"最小矾花面积: {min_floc_area_val} 像素")
    print(f"矾花密度: {floc_density:.6f} 个/像素²")

    # 8. 可视化
    result_img = cropped_img.copy()
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_floc_area and area < max_floc_area:
             mask = (labels == i).astype(np.uint8)
             result_img[mask == 1] = [0, 255, 0] 

    titles = ['Original Cropped', 'Sharpness Map (Laplacian)', 'Solid Mask (Filled)', 'Detected Flocs']
    images = [
        cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), 
        sharpness_map, 
        solid_mask, 
        cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, ax in enumerate(axes):
        if len(images[i].shape) == 3:
             ax.imshow(images[i])
        else:
             ax.imshow(images[i], cmap='gray', vmin=0, vmax=255) 
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

    # 返回这4个特征，方便后续接入 CSV 批量生成代码
    return floc_count, max_floc_area_val, min_floc_area_val, floc_density

if __name__ == "__main__":
    # 请替换为测试图像路径
    IMAGE_FILE = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_folders\normal\14_20.6_12KG\frame_1700.jpg"
    
    analyze_flocs_by_sharpness(IMAGE_FILE, crop_ratio=0.7, min_floc_area=5, max_floc_area=300, edge_thresh=25)