import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 自定义数据集类 
# ==========================================
class FlocDataset(Dataset):
    def __init__(self, csv_file, scaler=None, is_train=True):
        # 读取数据
        self.data = pd.read_csv(csv_file)
        
        # 提取特征 
        feature_cols = [
            'raw_turbidity',   
            'raw_temperature', 
            'floc_count', 
            'max_floc_area', 
            'min_floc_area', 
            'floc_density'
        ]
        X = self.data[feature_cols].values
        
        # 提取标签
        self.y = self.data['label'].values
        
        # 数据标准化 
        if is_train:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler # 验证/测试集必须使用训练集的 scaler
            self.X = self.scaler.transform(X)
            
        # 转换为 PyTorch 张量
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. 定义 MLP 神经网络架构 
# ==========================================
class FlocMLP(nn.Module):

    def __init__(self, input_dim=6, num_classes=3):
        super(FlocMLP, self).__init__()
        
        # 定义网络层：输入6 -> 隐藏层32 -> 隐藏层16 -> 输出3
        self.network = nn.Sequential(
            
            # --- 第 1 个隐藏层 ---
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),       # 批归一化，加速收敛
            nn.ReLU(),
            nn.Dropout(0.2),          # Dropout 防止过拟合
            
            # --- 第 2 个隐藏层 ---
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, num_classes) 
            # 注意：PyTorch的交叉熵损失函数自带Softmax，所以最后不需要加Softmax层
        )
        
    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. 核心训练流程
# ==========================================
def train_model(train_csv, val_csv, epochs=50, batch_size=32, lr=0.001):
    # 检测是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")
    
    # 1. 准备数据
    train_dataset = FlocDataset(train_csv, is_train=True)
    val_dataset = FlocDataset(val_csv, scaler=train_dataset.scaler, is_train=False) # 使用相同的scaler
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. 初始化模型、损失函数和优化器 
    model = FlocMLP(input_dim=6, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. 开始训练循环
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()       # 梯度清零
            outputs = model(batch_X)    # 前向传播
            loss = criterion(outputs, batch_y) # 计算损失
            loss.backward()             # 反向传播
            optimizer.step()            # 更新权重
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
            
        train_acc = correct_train / total_train
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # 验证阶段不计算梯度，节省内存
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                
        val_acc = correct_val / total_val
        
        # 打印本轮日志
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss/total_train:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss/total_val:.4f}, Val Acc: {val_acc:.4f}")

    print("\n✅ 训练完成！")
    # 保存训练好的模型权重
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 将时间戳、超参数和最终的验证集准确率(保留两位小数)组合成文件名
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")

    model_name = f"floc_mlp_acc{val_acc:.2f}_lr{lr}_bs{batch_size}_{current_time}.pth"
    model_path = os.path.join(save_dir, model_name)

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")

# ================== 运行代码 ==================
if __name__ == "__main__":
    #训练集和验证集路径
    TRAIN_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\train_set.csv"
    VAL_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\val_set.csv"
    
    # 开始训练
    train_model(TRAIN_CSV, VAL_CSV, epochs=50, batch_size=32, lr=0.001)