import os
# 允许 OpenMP 库重复加载，防止运行崩溃
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. 重建 CNN 模型架构 (必须与训练时完全一致)
# ==========================================
class FlocCNN(nn.Module):
    def __init__(self, input_features=6, num_classes=3):
        super(FlocCNN, self).__init__()
        
        # 卷积特征提取模块
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        # 分类器模块
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * input_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. 核心评估函数
# ==========================================
def evaluate_model_on_test_set(test_csv, train_csv, model_path):
    print("正在加载数据与 CNN 模型，准备进行最终评估...\n")
    
    feature_cols = [
        'raw_turbidity', 
        'raw_temperature', 
        'floc_count', 
        'max_floc_area', 
        'min_floc_area', 
        'floc_density'
    ]
    
    train_df = pd.read_csv(train_csv)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    
    test_df = pd.read_csv(test_csv)
    X_test_raw = test_df[feature_cols].values
    y_test_numpy = test_df['label'].values
    X_test_scaled = scaler.transform(X_test_raw)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 【核心修改】：为 1D-CNN 增加 Channel 维度 (unsqueeze(1))
    X_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1).to(device)
    y_tensor = torch.LongTensor(y_test_numpy).to(device)
    
    # 实例化 CNN 模型并加载权重
    model = FlocCNN(input_features=6, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)
        
    predictions_numpy = predictions.cpu().numpy()
    
    # ==========================================
    # 3. 打印评估报告
    # ==========================================
    acc = accuracy_score(y_test_numpy, predictions_numpy)
    cm = confusion_matrix(y_test_numpy, predictions_numpy)
    
    acc_normal = cm[0][0] / sum(cm[0]) if sum(cm[0]) > 0 else 0
    acc_excessive = cm[1][1] / sum(cm[1]) if sum(cm[1]) > 0 else 0
    acc_insufficient = cm[2][2] / sum(cm[2]) if sum(cm[2]) > 0 else 0
    
    print("="*50)
    print("测试集评估结果 (1D-CNN):")
    print("="*50)
    print(f"测试集总数据量: {len(test_df)} 条")
    print(f"总体预测准确率: {acc * 100:.2f}%\n")
    
    print("各类别准确率:")
    print(f"   [正常 (0)] 准确率: {acc_normal * 100:.2f}%  (正确判断: {cm[0][0]} / 实际总数: {sum(cm[0])})")
    print(f"   [过多 (1)] 准确率: {acc_excessive * 100:.2f}%  (正确判断: {cm[1][1]} / 实际总数: {sum(cm[1])})")
    print(f"   [过少 (2)] 准确率: {acc_insufficient * 100:.2f}%  (正确判断: {cm[2][2]} / 实际总数: {sum(cm[2])})\n")
    
    print("混淆矩阵:")
    print(f"{'':>16} |{'预测正常':>6}|{'预测过多':>6}|{'预测过少':>6}|")
    print("-" * 50)
    print(f"{'真实正常 (0)':>12} | {cm[0][0]:>8} | {cm[0][1]:>8} | {cm[0][2]:>8} |")
    print(f"{'真实过多 (1)':>12} | {cm[1][0]:>8} | {cm[1][1]:>8} | {cm[1][2]:>8} |")
    print(f"{'真实过少 (2)':>12} | {cm[2][0]:>8} | {cm[2][1]:>8} | {cm[2][2]:>8} |")
    print("="*50)

# ================== 运行测试 ==================
if __name__ == "__main__":
    # 替换为你实际的测试集和训练集路径
    TEST_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\test_set.csv"
    TRAIN_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\train_set.csv"
    
    # 替换为最新训练出来的 1D-CNN 模型路径 (.pth)
    SAVED_MODEL = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\saved_models\floc_cnn_acc0.86_lr0.001_bs32_0412_2029.pth"
    
    evaluate_model_on_test_set(TEST_CSV, TRAIN_CSV, SAVED_MODEL)