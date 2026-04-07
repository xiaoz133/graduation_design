import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 重建模型架构 (6 维输入)
# ==========================================
class FlocMLP(nn.Module):
    def __init__(self, input_dim=6, num_classes=3):
        super(FlocMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes) 
        )
    def forward(self, x):
        return self.network(x)

# ==========================================
# 2. 错例提取与保存函数
# ==========================================
def extract_bad_cases(test_csv, train_csv, model_path, original_dataset_root, output_error_dir):
    print("Bad Case Analysis...\n")
    
    # 定义标签映射，用于重建原图路径和生成中文文件夹
    label_map_reverse = {0: "normal", 1: "excessive", 2: "insufficient"}
    label_name_cn = {0: "正常", 1: "过多", 2: "过少"}

    # 1. 恢复 StandardScaler
    feature_cols = ['raw_turbidity', 'raw_temperature', 'floc_count', 'max_floc_area', 'min_floc_area', 'floc_density']
    train_df = pd.read_csv(train_csv)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    
    # 2. 读取测试集并处理
    test_df = pd.read_csv(test_csv)
    X_test_raw = test_df[feature_cols].values
    y_test_numpy = test_df['label'].values
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 3. 加载模型推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.FloatTensor(X_test_scaled).to(device)
    model = FlocMLP(input_dim=6, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)
    predictions_numpy = predictions.cpu().numpy()
    
    # 4. 开始对比并拷贝错例
    error_count = 0
    if not os.path.exists(output_error_dir):
        os.makedirs(output_error_dir)

    for idx, row in test_df.iterrows():
        true_label = int(row['label'])
        pred_label = int(predictions_numpy[idx])
        
        # 如果预测错误
        if true_label != pred_label:
            error_count += 1
            
            # 还原这张图片在原始数据集里的真实路径
            # 路径结构：DATASET_ROOT / 类别名(normal等) / 子批次(14_20.6_20KG等) / 图片名
            category_folder = label_map_reverse[true_label]
            subfolder = str(row['subfolder_info'])
            img_name = str(row['image_name'])
            
            original_img_path = os.path.join(original_dataset_root, category_folder, subfolder, img_name)
            
            # 如果原图存在，则进行拷贝
            if os.path.exists(original_img_path):
                # 建立类似 "True正常_Pred过多" 的子文件夹
                error_subfolder_name = f"True{label_name_cn[true_label]}_Pred{label_name_cn[pred_label]}"
                error_subfolder_path = os.path.join(output_error_dir, error_subfolder_name)
                
                if not os.path.exists(error_subfolder_path):
                    os.makedirs(error_subfolder_path)
                
                # 为了防止不同批次的同名图片覆盖，给拷贝过去的文件加上批次前缀
                new_img_name = f"{subfolder}_{img_name}"
                dest_path = os.path.join(error_subfolder_path, new_img_name)
                
                shutil.copy2(original_img_path, dest_path)
            else:
                print(f"⚠️ 警告: 找不到原始图片文件 -> {original_img_path}")

    print("="*50)
    print(f"✅ 错例提取完成！")
    print(f"总计检测测试集: {len(test_df)} 张图片")
    print(f"共发现误判错例: {error_count} 张图片")
    print(f"错例已按错误类型分类保存至: {output_error_dir}")
    print("="*50)

# ================== 运行代码 ==================
if __name__ == "__main__":
    # 1. 你的测试集 CSV 和 训练集 CSV
    TEST_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experience\dataset_splits\test_set.csv"
    TRAIN_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experience\dataset_splits\train_set.csv"
    
    # 2. 你最新跑出来的模型权重路径
    SAVED_MODEL = r"C:\Users\94508\Desktop\zds\graduation_design\experience\saved_models\floc_mlp_acc0.76_lr0.0001_bs32_0407_1706.pth"
    
    # 3. 最早构建数据集时的原始图片总目录 (为了能回去找到原图)
    ORIGINAL_DATASET_ROOT = r"C:\Users\94508\Desktop\zds\graduation_design\experience\dataset_folders"
    
    # 4. 把挑出来的错例保存
    OUTPUT_ERROR_DIR = r"C:\Users\94508\Desktop\zds\graduation_design\experience\bad_cases_analysis"
    
    extract_bad_cases(TEST_CSV, TRAIN_CSV, SAVED_MODEL, ORIGINAL_DATASET_ROOT, OUTPUT_ERROR_DIR)