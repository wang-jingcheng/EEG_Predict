import os
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import torch
from model import Predictor as Net
from util.conf import *
from scipy.stats import pearsonr

model = 'cnn3'
dataset = '(1)'
seed = 1

data_path = './test_data'
model_path = './saved/best_model_%s_%s_%d.pt'%(model, dataset, seed)

model = Net(INPUT_CHANNELS, HIDDEN_CHANNELS, OUTPUT_CHANNELS, model = model)
model.load_state_dict(torch.load(model_path))

inuputs, labels, preds = [], [], []
all_correlation = []

for filename in os.listdir(data_path):
    if filename.endswith('NT41_ZJQ_R_01.csv'):
        file_path = os.path.join(data_path, filename)
        df = pd.read_csv(file_path)

        x_data = df.iloc[INPUT_INDICES].to_numpy()
        y_data = df.drop(index=INPUT_INDICES).to_numpy()
        # x_data = np.expand_dims(x_data, axis=0) #仅针对二维卷积模型

        input = torch.tensor(x_data, dtype=torch.float32)
        input = input.unsqueeze(0)
        
        inuputs.append(input)
        labels.append(y_data)

for i in range(len(inuputs)):
    input = inuputs[i]
    label = labels[i]
    
    model.eval()
    with torch.no_grad():
        pred = model(input)
        pred = pred.numpy()
        pred = pred.squeeze(axis=0)
        preds.append(pred)

        # pred = pred.squeeze(axis=0) #仅针对二维卷积模型

id = 0
time_start = 1000
time_end = time_start + 1000
    
for i in range(OUTPUT_CHANNELS):
    correlation, _ = pearsonr(labels[id][i,:], preds[id][i,:])
    all_correlation.append(correlation)
mean_correlation = np.mean(all_correlation).astype(float)

x0 = range(OUTPUT_CHANNELS)
plt.figure(figsize=(10, 5))
plt.title("Correlation of Predicted and Real Data\nMean correlation coefficient:%f"%(mean_correlation))
plt.grid(ls="--", alpha=0.5)
plt.bar(x0, all_correlation)
plt.axhline(mean_correlation,color='red',linestyle='--')
plt.axhline(0.9,color='red')
plt.xlabel('Channel_id')
plt.ylabel('Correlation')
plt.show()

plt.figure(figsize=(20, 200))
for chanel_id in range(OUTPUT_CHANNELS):
    x = range(time_start,time_end)
    y1 = preds[id][chanel_id,time_start:time_end]
    y2 = labels[id][chanel_id,time_start:time_end]

    plt.subplot(OUTPUT_CHANNELS,1,chanel_id+1)
    plt.plot(x, y1, label='pred')
    plt.plot(x, y2, label='real')
    plt.xlabel('Time')
    plt.ylabel('value')
    plt.title('Predicted and actual data for channel:%d'%(chanel_id))
    plt.legend()

plt.show()
