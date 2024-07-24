import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from util.conf import *
# 指定数据集文件夹路径
file_path = DIRECTORY

#加载数据集
def process_csv_files(directory,x_indices=[5, 12, 14, 21, 40, 49, 51, 58]):  
    x_data_list = []
    y_data_list = []
    
    # 遍历目录中的所有CSV文件
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 确保数据行数为64行
            if df.shape[0] != 64:
                raise ValueError(f"CSV文件 {filename} 应包含64行数据")
            
            x_data = df.iloc[x_indices].to_numpy()
            y_data = df.drop(index=x_indices).to_numpy()
            
            x_data_list.append(x_data)
            y_data_list.append(y_data)
    
    # 将列表转换为三维NumPy数组
    x_data_array = np.array(x_data_list)
    y_data_array = np.array(y_data_list)
    
    return x_data_array, y_data_array

x_data, y_data = process_csv_files(file_path)
print("x_data:")
print(x_data.shape)
print("\ny_data:")
print(y_data.shape)

#定义数据预处理函数
def split_and_convert_to_tensor(x_data, y_data, test_size=0.1):
    # 分割数据集为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)
    
    # 转换为PyTorch张量
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor

x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor = split_and_convert_to_tensor(x_data, y_data)

# 创建数据集和数据加载器
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

