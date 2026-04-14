import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 自定义数据集类 (已适配 1D-CNN)
# ==========================================
class FlocDataset(Dataset):
    def __init__(self, csv_file, scaler=None, is_train=True):
        self.data = pd.read_csv(csv_file)
        
        feature_cols = [
            'raw_turbidity',   
            'raw_temperature', 
            'floc_count', 
            'max_floc_area', 
            'min_floc_area', 
            'floc_density'
        ]
        X = self.data[feature_cols].values
        self.y = self.data['label'].values
        
        if is_train:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler 
            self.X = self.scaler.transform(X)
            
        # 核心修改：为 1D-CNN 增加 Channel 维度
        # 原始形状 (N, 6) -> 转换后形状 (N, 1, 6)
        self.X = torch.FloatTensor(self.X).unsqueeze(1)
        self.y = torch.LongTensor(self.y)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. 定义 1D-CNN 神经网络架构
# ==========================================
class FlocCNN(nn.Module):
    def __init__(self, input_features=6, num_classes=3):
        super(FlocCNN, self).__init__()
        
        # 卷积特征提取模块
        self.features = nn.Sequential(
            # 输入通道数为1，输出通道数为16，卷积核大小为3
            # 输入形状: (Batch, 1, 6) -> 输出形状: (Batch, 16, 6) (因为 padding=1)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            # 第二层卷积
            # 输入形状: (Batch, 16, 6) -> 输出形状: (Batch, 32, 6)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        # 分类器模块
        # 经过两层卷积后，特征图大小为 32通道 * 6长度 = 192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * input_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3), # 稍微提高 Dropout 比例以应对 CNN 强大的拟合能力
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x 的形状必须是 (Batch, Channels, Length)
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 3. 核心训练流程
# ==========================================
def train_model(train_csv, val_csv, epochs=50, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")
    
    train_dataset = FlocDataset(train_csv, is_train=True)
    val_dataset = FlocDataset(val_csv, scaler=train_dataset.scaler, is_train=False) 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 替换为新的 CNN 模型
    model = FlocCNN(input_features=6, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()       
            outputs = model(batch_X)    
            loss = criterion(outputs, batch_y) 
            loss.backward()             
            optimizer.step()            
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
            
        train_acc = correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): 
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                
        val_acc = correct_val / total_val
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss/total_train:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss/total_val:.4f}, Val Acc: {val_acc:.4f}")

    print("\n✅ 训练完成！")
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    model_name = f"floc_cnn_acc{val_acc:.2f}_lr{lr}_bs{batch_size}_{current_time}.pth"
    model_path = os.path.join(save_dir, model_name)

    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")

# ================== 运行代码 ==================
if __name__ == "__main__":
    TRAIN_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\train_set.csv"
    VAL_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\val_set.csv"
    
    train_model(TRAIN_CSV, VAL_CSV, epochs=50, batch_size=32, lr=0.001)