import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from util.conf import *
from tqdm import tqdm
from model import Predictor as Net
from scipy.stats import pearsonr

def preprocess_data(data, crop_size):
    """预处理数据，包括随机裁剪和标准化"""
    # 随机裁剪
    original_size = data.shape[1]
    start_idx = np.random.randint(0, original_size - crop_size + 1)
    cropped_data = data[:, start_idx:start_idx + crop_size]
    
    # 使用StandardScaler进行标准化
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(cropped_data)
    
    return cropped_data

#定义加载数据集函数
def DataLoaderGenerator(directory, input_indices, test_size):
    x_data_list = []
    y_data_list = []

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
             # 确保数据行数为64行
            if df.shape[0] != 64:
                raise ValueError(f"CSV文件 {filename} 应包含64行数据")  
            
            for i in range(10):
                df = preprocess_data(df.to_numpy(), 8192)
                df = pd.DataFrame(df)
            
                x_data = df.iloc[input_indices].to_numpy()
                y_data = df.drop(index=input_indices).to_numpy()

                x_data_list.append(x_data)
                y_data_list.append(y_data)
            
    
    # 将列表转换为三维NumPy数组
    x_data_array = np.array(x_data_list)
    y_data_array = np.array(y_data_list)
    
    # 三维数据转换为四维数据，用于2D卷积
    # x_data_array = np.expand_dims(x_data_array, axis=1)
    # y_data_array = np.expand_dims(y_data_array, axis=1)

    # 分割数据集为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_data_array, y_data_array, test_size=test_size, random_state=42)
    
    print("-----------------------------------------")
    print("训练集(x_train)的大小:",x_train.shape)
    print("测试集(x_test)的大小:",x_test.shape)
    print("-----------------------------------------")

    # 转换为PyTorch张量
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    #创建训练集和测试集
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset

# 训练模型的函数
def train_model(model, device, train_loader, criterion, optimizer, clip):
    model.train()
    epoch_loss = 0.0
    epoch_correlation_score = 0.0
    train_loss = 0.0
    train_correlation_score = 0.0
    cnt = 0
    for i, batch in tqdm(enumerate(train_loader)):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.cpu()
        labels = labels.cpu()

        loss = criterion(outputs, labels)
        loss.backward()
        
        #梯度裁剪与爆炸检测
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        update = True
        for pp in model.parameters():
            if pp.requires_grad and pp.grad is not None:
                if torch.isnan(pp.grad).any():
                    update=False
                    break
        if update:
            optimizer.step()
        else:
            print('Gradient explosion and discard this batch!!!')

        y_trues = labels.numpy()
        y_preds = outputs.detach().numpy()
        correlation_score = correlation(y_trues, y_preds)
        
        epoch_correlation_score += correlation_score
        epoch_loss += loss.item()
        cnt = i
    train_loss = epoch_loss / (cnt+1)
    train_correlation_score = epoch_correlation_score / (cnt+1)
    return train_loss, train_correlation_score

# 评估模型的函数
def evaluate_model(model, device, test_loader, criterion):
    model.eval()
    epoch_loss = 0.0
    epoch_correlation_score = 0.0
    test_loss = 0.0
    test_correlation_score = 0.0
    cnt = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs.cpu()
            labels = labels.cpu()
                
            loss = criterion(outputs, labels)

            y_trues = labels.numpy()
            y_preds = outputs.detach().numpy()
            correlation_score = correlation(y_trues, y_preds)

            epoch_correlation_score += correlation_score
            epoch_loss += loss.item()
            cnt = i
    test_loss = epoch_loss / (cnt+1)
    test_correlation_score = epoch_correlation_score / (cnt+1)
    return test_loss, test_correlation_score

# 参数量统计函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 初始化权重函数
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)
        
# 时间统计函数
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 计算相关系数函数
def correlation(y_trues, y_pred):
    all_correlation = []
    for i in range(BATCH_SIZE):
        for j in range(OUTPUT_CHANNELS):
            correlation, _ = pearsonr(y_trues[i,j,:], y_pred[i,j,:])
            all_correlation.append(correlation)
    return np.mean(all_correlation)

# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.determinsitic=True


def run(model, device, train_dataset, test_dataset, total_epoch, best_loss, optimizer, criterion, clip):
    losses = {'train':[], 'val':[]}
    acces = {'train':[], 'val':[]}
    for step in range(total_epoch):
        print('Epoch {} / {}:'.format(step + 1, total_epoch))
        start_time = time.time()
        print('Training...')
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loss, train_acc = train_model(model, device, train_loader, criterion, optimizer, clip)
        print('Evaluating...')
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loss, val_acc = evaluate_model(model, device, test_loader, criterion)
        end_time = time.time()

        if step > WARMUP:
            scheduler.step(val_loss)

        losses['train'].append(train_loss)
        losses['val'].append(val_loss)
        acces['train'].append(train_acc)
        acces['val'].append(val_acc)

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './saved/best_model_%s_%s_%d.pt'%(MODEL, DATASET, SEED))
        torch.save(model.state_dict(), './saved/latest_model_%s_%s_%d.pt'%(MODEL, DATASET, SEED))
        
        with open('result/result_%s_%s_%d.json'%(MODEL, DATASET, SEED), 'w') as f:
            json.dump({'loss':losses, 'acc':acces}, f)
        
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}')
        print(f'\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}')


if __name__ == '__main__':
    setup_seed(SEED)

    print('Loading dataset...')
    train_dataset, test_dataset= DataLoaderGenerator(DIRECTORY, INPUT_INDICES, TEST_SIZE)
    
    model = Net(INPUT_CHANNELS, HIDDEN_CHANNELS, OUTPUT_CHANNELS, model = MODEL)
    print(f'The model has {count_parameters(model):,} trainable parameters in total')
    
    model.apply(initialize_weights)
    model = model.to(DEVICE)
  
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY,eps=ADAM_EPS)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=FACTOR, patience=PATIENCE)
    criterion = nn.MSELoss()

    run(model=model, device=DEVICE, train_dataset=train_dataset, test_dataset=test_dataset, total_epoch=EPOCH, best_loss=INF, 
        optimizer=optimizer, criterion=criterion, clip=CLIP)