import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(csv_path, output_dir):
    # 1. 读取构建的完整数据集
    print(f"正在读取数据集: {csv_path}")
    df = pd.read_csv(csv_path)
    total_len = len(df)
    
    # 2. 第一次切分：分离出 70% 的训练集，剩下 30% 作为临时集 (验证+测试)
    # stratify=df['label'] 确保各类标签比例均衡
    df_train, df_temp = train_test_split(
        df, 
        test_size=0.30, 
        random_state=42,       # 设定随机种子，保证每次运行切分结果一样，方便复现
        stratify=df['label']   # 核心：按标签比例分层抽样
    )
    
    # 3. 第二次切分：将剩下的 30% 对半分，得到 15% 验证集和 15% 测试集
    df_val, df_test = train_test_split(
        df_temp, 
        test_size=0.50, 
        random_state=42, 
        stratify=df_temp['label']
    )
    
    # 4. 打印切分结果验证
    print("\n--- 数据集划分完成 ---")
    print(f"总数据量: {total_len} 条")
    print(f"训练集 (Train): {len(df_train)} 条 ({len(df_train)/total_len*100:.1f}%)")
    print(f"验证集 (Val):   {len(df_val)} 条 ({len(df_val)/total_len*100:.1f}%)")
    print(f"测试集 (Test):  {len(df_test)} 条 ({len(df_test)/total_len*100:.1f}%)")
    
    # 5. 保存切分后的数据集到本地
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_train.to_csv(os.path.join(output_dir, 'train_set.csv'), index=False)
    df_val.to_csv(os.path.join(output_dir, 'val_set.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'test_set.csv'), index=False)
    
    print(f"\n✅ 切分后的文件已保存至: {output_dir}")


if __name__ == "__main__":
    # 总表路径
    INPUT_CSV = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\floc_multimodal_dataset.csv"
    
    # 划分后的输出目录
    OUTPUT_FOLDER = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits"
    
    split_dataset(INPUT_CSV, OUTPUT_FOLDER)