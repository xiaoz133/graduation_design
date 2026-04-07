import cv2

import numpy as np

import matplotlib.pyplot as plt



def analyze_flocs_ternary(image_path, crop_ratio=0.4, min_floc_area=5, max_floc_area=200, t_low=90, t_high=190):

    """
    使用三值化方法分析矾花图像：过滤焦外模糊背景，仅提取前景高亮颗粒。
    :param image_path: 图片路径
    :param crop_ratio: 中心截取比例
    :param min_floc_area: 最小有效面积阈值（过滤微小噪点）
    :param max_floc_area: 最大有效面积阈值（过滤过大颗粒）
    :param t_low: 低阈值（区分水体背景和焦外矾花）
    :param t_high: 高阈值（区分焦外矾花和焦内清晰矾花）
    """

    # 1. 读取与截取
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"错误：无法读取图片 {image_path}")
        return


    height, width = original_img.shape[:2]
    crop_h, crop_w = int(height * crop_ratio), int(width * crop_ratio)
    y1, x1 = int((height - crop_h) / 2), int((width - crop_w) / 2)
    cropped_img = original_img[y1:y1+crop_h, x1:x1+crop_w]


    # 2. 灰度化
    gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)   

    # 3. 三值化处理 (Ternarization)
    # 创建一个全黑的图像作为底板
    ternary_img = np.zeros_like(gray_cropped, dtype=np.uint8)   

    # 处于低阈值和高阈值之间的部分，赋值为灰色 (128) -> 焦外模糊矾花
    ternary_img[(gray_cropped > t_low) & (gray_cropped <= t_high)] = 128   

    # 大于高阈值的部分，赋值为白色 (255) -> 焦内清晰矾花
    ternary_img[gray_cropped > t_high] = 255

    # 4. 生成仅包含白色前景的二值掩码，用于特征提取
    foreground_mask = (ternary_img == 255).astype(np.uint8) * 255

    # 5. 特征提取 (仅针对白色掩码)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground_mask, connectivity=8)

    valid_floc_areas = []

    # 从 1 开始，跳过背景 (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_floc_area and area < max_floc_area:  # 过滤过小和过大的颗粒
            valid_floc_areas.append(area)


    floc_count = len(valid_floc_areas)


    print(f"\n--- 前景矾花分析结果 (中心 {crop_ratio*100}%) ---")
    print(f"有效前景矾花数量: {floc_count}")


    if floc_count > 0:

        max_floc_area_val = max(valid_floc_areas)
        min_floc_area_val = min(valid_floc_areas)
        print(f"最大颗粒面积: {max_floc_area_val} 像素")
        print(f"最小颗粒面积: {min_floc_area_val} 像素")

    else:

        print(f"未检测到有效的前景矾花。")


    cropped_area_px = crop_w * crop_h
    floc_density = floc_count / cropped_area_px

    print(f"前景矾花密度: {floc_density:.6f} 个/像素²")

    # 6. 结果可视化
    # 用彩色标记识别出的有效前景颗粒，方便观察

    result_img = cropped_img.copy()

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_floc_area and area < max_floc_area:

             # 用绿色半透明遮罩标记检测到的矾花
             mask = (labels == i).astype(np.uint8)

             result_img[mask == 1] = [0, 255, 0] # BGR格式的绿色

    # 合并显示图像
    result_display = cv2.addWeighted(cropped_img, 0.6, result_img, 0.4, 0)

    # 将标题改为彩色裁剪图
    titles = ['Cropped Image (Color)', 'Ternary Map (Black/Gray/White)', 'Foreground Mask (White Only)', 'Detected Flocs (Green)']

    # 将第一个图像替换为转换为 RGB 格式的 cropped_img

    images = [

        cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB),
        ternary_img,
        foreground_mask,
        cv2.cvtColor(result_display, cv2.COLOR_BGR2RGB)

    ]

   

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, ax in enumerate(axes):

        # 动态判断图像通道数：彩色图 (3通道) 直接显示，单通道图强制使用黑白灰度映射
        if len(images[i].shape) == 3:

             ax.imshow(images[i])

        else:
             ax.imshow(images[i], cmap='gray', vmin=0, vmax=255)

        ax.set_title(titles[i])

        ax.axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    # 图片路径
    IMAGE_FILE = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_folders\insufficient\14_20.6_7KG\frame_1585.jpg"
    
    analyze_flocs_ternary(IMAGE_FILE, crop_ratio=0.4, min_floc_area=5, max_floc_area=300, t_low=90, t_high=210)