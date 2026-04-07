import os
import pandas as pd
import cv2
import numpy as np

# ==========================================
# 1. 特征提取函数
# ==========================================
def extract_features(image_path, crop_ratio=0.7, min_floc_area=5, max_floc_area=300, edge_thresh=25):
    """
    特征提取函数（基于边缘清晰度）：
    从单张图片中提取真正处于焦平面内的矾花数量、最大面积、最小面积和密度特征。
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None 

    height, width = original_img.shape[:2]
    crop_h, crop_w = int(height * crop_ratio), int(width * crop_ratio)
    y1, x1 = int((height - crop_h) / 2), int((width - crop_w) / 2)
    cropped_img = original_img[y1:y1+crop_h, x1:x1+crop_w]

    gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_cropped, (3, 3), 0)
    
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    sharpness_map = cv2.convertScaleAbs(laplacian)
    
    _, edge_mask = cv2.threshold(sharpness_map, edge_thresh, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(gray_cropped)
    cv2.drawContours(solid_mask, contours, -1, 255, thickness=cv2.FILLED)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(solid_mask, connectivity=8)

    valid_floc_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_floc_area < area < max_floc_area:
            valid_floc_areas.append(area)

    floc_count = len(valid_floc_areas)
    cropped_area_px = crop_w * crop_h
    floc_density = floc_count / cropped_area_px if cropped_area_px > 0 else 0

    if floc_count > 0:
        max_floc_area_val = max(valid_floc_areas)
        min_floc_area_val = min(valid_floc_areas)
    else:
        max_floc_area_val = 0
        min_floc_area_val = 0

    return floc_count, max_floc_area_val, min_floc_area_val, floc_density


# ==========================================
# 2. 嵌套文件夹批处理 (新增了理化参数解析)
# ==========================================
def build_dataset_nested(root_dir, output_csv):
    dataset_records = []
    
    label_map = {
        "normal": 0,       
        "excessive": 1,    
        "insufficient": 2  
    }
    
    total_images = 0
    
    for category_name in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_name)
        
        if os.path.isdir(category_path) and category_name in label_map:
            label_value = label_map[category_name]
            print(f"正在扫描顶层类别: [{category_name}] ...")
            
            for dirpath, dirnames, filenames in os.walk(category_path):
                
                # 获取当前所在的子文件夹名称 (例如: "14_20.6_20KG")
                subfolder_name = os.path.basename(dirpath)
                
               
                # 解析文件夹名称，提取浊度和温度
                turbidity = None
                temperature = None
                
                # 假设命名规则严格为 "浊度_温度_其他"
                parts = subfolder_name.split('_')
                if len(parts) >= 2:
                    try:
                        turbidity = float(parts[0])      # 提取 14
                        temperature = float(parts[1])    # 提取 20.6
                    except ValueError:
                        # 如果遇到无法转换的字符，打印警告并跳过或设为空
                        print(f"⚠️ 警告: 无法从文件夹名 '{subfolder_name}' 中解析理化参数。")
                
                # 如果这个文件夹没有正确的理化参数，为了保证数据纯净，可以选择跳过
                # 若跳过，可以取消下面这两行的注释：
                # if turbidity is None or temperature is None:
                #     continue

                for img_name in filenames:
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(dirpath, img_name)
                        
                        # 1. 提取图像特征
                        features = extract_features(img_path)
                        
                        if features is not None:
                            floc_count, max_area, min_area, density = features
                            
                            # 2. 组装数据记录 (加入了 raw_turbidity 和 raw_temperature)
                            record = {
                                "image_name": img_name,
                                "subfolder_info": subfolder_name, 
                                "raw_turbidity": turbidity,      # 新增特征：原水浊度
                                "raw_temperature": temperature,  # 新增特征：温度
                                "floc_count": floc_count,
                                "max_floc_area": max_area,
                                "min_floc_area": min_area,
                                "floc_density": density,
                                "label": label_value 
                            }
                            dataset_records.append(record)
                            total_images += 1
                            
                            if total_images % 1000 == 0:
                                print(f"已提取 {total_images} 张图片特征...")

    # 3. 保存为 CSV
    df = pd.DataFrame(dataset_records)
    
    # 稍微调整一下表格列的顺序，把标签放在最后，特征放中间，看起来更舒服
    columns_order = [
        "image_name", "subfolder_info", 
        "raw_turbidity", "raw_temperature", 
        "floc_count", "max_floc_area", "min_floc_area", "floc_density", 
        "label"
    ]
    df = df[columns_order]
    
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✅ 数据集构建完成！共提取 {total_images} 条有效数据。")
    print(f"表格已保存至: {output_csv}")
    print("\n--- 数据分布统计 ---")
    print(df['label'].value_counts().rename(index={0: 'Normal (0)', 1: 'Excessive (1)', 2: 'Insufficient (2)'}))

# ================== 运行批量处理 ==================
if __name__ == "__main__":
    # 将所有子文件夹都按照 "浊度_温度_加药量" 的格式重命名，例如 "14_20.6_20KG"，以便自动解析理化参数
    DATASET_ROOT = r"C:\Users\94508\Desktop\zds\graduation_design\experience\dataset_folders"
    OUTPUT_CSV_FILE = r"C:\Users\94508\Desktop\zds\graduation_design\experience\floc_multimodal_dataset.csv"
    
    build_dataset_nested(DATASET_ROOT, OUTPUT_CSV_FILE)