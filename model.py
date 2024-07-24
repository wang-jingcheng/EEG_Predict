import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet
import numpy as np
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, model = "cnn1"):
        super(Predictor, self).__init__()
        self.model = model
        # self.fc = nn.Linear(hidden_channels, output_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.output_linear = nn.Linear(input_channels, output_channels)

        if model == 'cnn1':
            self.predictor = CNN1(input_channels,hidden_channels, output_channels)
        elif model == 'cnn2':
            self.predictor = CNN2(input_channels, hidden_channels, output_channels)
        elif model == 'cnn3':
            self.predictor = CNN3(input_channels, hidden_channels, output_channels)
        elif model == 'cnn4':
            self.predictor = CNN4(input_channels, hidden_channels, output_channels)
        # elif model == 'gru':
        #     self.predictor = GRU(input_channels, hidden_channels, output_channels)    
        # elif model == 'lstm':
        #     self.predictor = LSTM(input_channels, hidden_channels, output_channels)
        # elif model == 'transformer':
        #     self.predictor = Transformer(input_channels, hidden_channels, output_channels)
    
    def forward(self, x):
        output = self.predictor(x)
        return output


#定义CNN1模型（三层卷积）
class CNN1(nn.Module):
    def __init__(self, input_channels,hidden_channels, output_channels):
        super(CNN1, self).__init__()
        self.features = nn.Sequential(
           nn.Conv1d(input_channels,hidden_channels,kernel_size=3,stride=1,padding=1),
           nn.BatchNorm1d(hidden_channels),
           nn.ReLU(),
           nn.Conv1d(hidden_channels,64,kernel_size=3,stride=1,padding=1),
           nn.BatchNorm1d(64),
           nn.ReLU(),
           nn.Conv1d(64,output_channels,kernel_size=3,stride=1,padding=1),
           )
    def forward(self, x):
        x = self.features(x)
        return x
    

#定义CNN2模型（两层卷积+全连接）    
"经测试,该模型在测试集上表现较差,loss不下降,不采用"
class CNN2(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, sequence_length=1024):
        super(CNN2, self).__init__()
        self.output_channels = output_channels

        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, out_channels=128, kernel_size=3, padding=1)

        pooled_sequence_length = sequence_length // 4
        self.fc1 = nn.Linear(128 * pooled_sequence_length, 1024) 
        self.fc2 = nn.Linear(1024, output_channels*sequence_length)
 
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        batch_size, input_chanels, sequence_length = input.shape
        x = x.view(batch_size, self.output_channels, sequence_length)
        return x
    
#定义CNN3模型（五层卷积）
"经测试,该模型模型训练速度较慢,50个epoch大约需要30多个小时,有几个通道训练效果极差,其他通道跟一维的cnn差不多,不采用"
class CNN3(nn.Module):
    def __init__(self, input_channels,hidden_channels, output_channels):
        super(CNN3, self).__init__()
        self.features = nn.Sequential(
           nn.Conv1d(in_channels=input_channels, out_channels=512,kernel_size=11,stride=1,padding=5),
           nn.BatchNorm1d(512),
           nn.ReLU(),
           nn.Conv1d(in_channels=512,out_channels=256,kernel_size=9,stride=1,padding=4),
           nn.BatchNorm1d(256),
           nn.ReLU(),
           nn.Conv1d(in_channels=256,out_channels=128,kernel_size=5,stride=1,padding=2),
           nn.BatchNorm1d(128),
           nn.ReLU(),
           nn.Conv1d(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0),
           nn.BatchNorm1d(128),
           nn.ReLU(),
           nn.Conv1d(in_channels=128,out_channels=output_channels,kernel_size=7,stride=1,padding=3),
           )
    def forward(self, x):
        x = self.features(x)
        return x

#定义CNN4模型（二维卷积）
class CNN4(nn.Module):
    def __init__(self, input_channels,hidden_channels, output_channels):
        super(CNN4, self).__init__()
        self.features = nn.Sequential(
           nn.Conv2d(in_channels=input_channels,out_channels=128,kernel_size=11,stride=1,padding=(29,5)),
           nn.BatchNorm2d(128),
           nn.ReLU(),
           nn.Conv2d(in_channels=128,out_channels=64,kernel_size=9,stride=1,padding=4),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5,stride=1,padding=2),
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,stride=1,padding=0),
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.Conv2d(in_channels=32,out_channels=output_channels,kernel_size=7,stride=1,padding=3),
           )
    def forward(self, x):
        x = self.features(x)
        return x

# #定义经典GRU模型
# class GRU(nn.Module):
#     def __init__(self, d_model, d_input):
#         super().__init__() 
#         self.d_model=d_model
#         self.d_input=d_input
#         # self.fc1 = nn.Linear(d_model, d_model)
#         # self.relu = nn.ReLU(inplace=True)
#         # self.img_trans = ImgTransform(d_model, block_size, DATASET)
#         self.gru = nn.GRU(input_size=d_input, hidden_size=d_model, batch_first=True)
#         # self.output_linear = nn.Linear(d_model, output_dim)

#     def forward(self, x):
#         # x = self.img_trans(x)
#         # x = self.fc1(x)
#         # x = self.relu(x)
#         output, _ = self.gru(x)
#         # output = self.output_linear.forward(output[:,-1,:])
#         # y = F.softmax(output)
#         return output[:,-1,:]

#定义LSTM模型           
# class LSTM(nn.Module):
#     def __init__(self, input_channels, hidden_channels, output_channels):
#         super(LSTM, self).__init__()
#         self.hidden_channels = hidden_channels
#         self.num_layers = 2
#         self.lstm = nn.LSTM(input_channels, hidden_channels, 2, batch_first=True)
#         self.fc = nn.Linear(hidden_channels, output_channels)
    
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_channels)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_channels)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out

# #定义Transformer模型
# class Transformer(nn.Module):
#     def __init__(self, d_model, d_input, num_heads=N_HEADS, num_layers=N_ENC_LAYERS):
#         super().__init__()
#         self.d_model = d_model
#         self.d_input = d_input
       
#         #线性变换层
#         # self.fc1 = nn.Linear(d_model*self.ratio**2,  d_model)
#         # # 嵌入层
#         # self.embedding = nn.Linear(input_dim, hidden_dim)
#         # Transformer编码器层
#         # self.img_trans = ImgTransform(d_model, block_size, DATASET)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_input, nhead=num_heads, dim_feedforward=d_model)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         # 线性分类层
#         # self.fc2 = nn.Linear(d_model, output_dim)
#         def forward(self, x):
#         # x = self.img_trans(x)
#         # x=self.fc1(x)
#         # x = self.embedding(x)
#             x = x.permute(1, 0, 2)  # 调整维度顺序
#             output = self.transformer_encoder(x)
#         # output = encoded.mean(dim=0)  # 平均池化
#         # output = self.fc2(output)
#         # output = F.softmax(output)
        
#         return output[-1,:,:]

# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward, dropout):
#         super(TransformerModel, self).__init__()
#         self.transformer = nn.Transformer(
#             d_model=input_dim,
#             nhead=nhead,
#             num_encoder_layers=num_encoder_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout
#         )
#         self.fc = nn.Linear(input_dim, output_dim)

#     def forward(self, src):
#         # Transformer expects input of shape (seq_len, batch_size, feature_size)
#         src = src.permute(1, 0, 2)
#         transformer_out = self.transformer(src)
#         out = self.fc(transformer_out)
#         # Convert back to (batch_size, seq_len, feature_size)
#         out = out.permute(1, 0, 2)
#         return out
