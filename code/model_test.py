import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. 重建模型架构
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
# 2. 核心评估函数
# ==========================================
def evaluate_model_on_test_set(test_csv, train_csv, model_path):
    print("正在加载数据与模型，准备进行最终评估...\n")
    
    # 1. 明确我们要使用的 6 个特征
    feature_cols = [
        'raw_turbidity', 
        'raw_temperature', 
        'floc_count', 
        'max_floc_area', 
        'min_floc_area', 
        'floc_density'
    ]
    
    # 2. 恢复 StandardScaler
    train_df = pd.read_csv(train_csv)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    
    # 3. 读取并处理测试集
    test_df = pd.read_csv(test_csv)
    X_test_raw = test_df[feature_cols].values
    y_test_numpy = test_df['label'].values
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 4. 转换为张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_tensor = torch.LongTensor(y_test_numpy).to(device)
    
    # 5. 加载模型
    model = FlocMLP(input_dim=6, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 6. 模型推理
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)
        
    predictions_numpy = predictions.cpu().numpy()
    
    # ==========================================
    # 3. 打印评估报告
    # ==========================================
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    # 计算总体准确率与混淆矩阵
    acc = accuracy_score(y_test_numpy, predictions_numpy)
    cm = confusion_matrix(y_test_numpy, predictions_numpy)
    
    # 通过混淆矩阵计算每个类别的独立准确率 (对角线数值 / 该行总数)
    # 加上 if 判断是为了防止某种类别在测试集中正好有 0 条数据导致除以 0 报错
    acc_normal = cm[0][0] / sum(cm[0]) if sum(cm[0]) > 0 else 0
    acc_excessive = cm[1][1] / sum(cm[1]) if sum(cm[1]) > 0 else 0
    acc_insufficient = cm[2][2] / sum(cm[2]) if sum(cm[2]) > 0 else 0
    
    print("="*50)
    print("测试集评估结果:")
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
    # 1. 测试集 CSV 路径 
    TEST_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\test_set.csv"
    
    # 2. 训练集 CSV 路径 (用于恢复 StandardScaler)
    TRAIN_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\train_set.csv"
    
    # 3. 最新训练出来的 6 维输入的模型路径
    SAVED_MODEL = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\saved_models\0.7\floc_mlp_acc0.80_lr0.001_bs32_0407_1746.pth"
    
    evaluate_model_on_test_set(TEST_CSV, TRAIN_CSV, SAVED_MODEL)