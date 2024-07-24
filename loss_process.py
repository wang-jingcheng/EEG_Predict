import matplotlib.pyplot as plt
import numpy as np
import json

model = 'cnn3'
dataset = '(1)'
seed = 1

with open('./result/result_%s_%s_%d.json'%(model, dataset, seed),'r') as f:
    result = json.load(f)

train_loss = np.array(result['loss']['train'])
val_loss = np.array(result['loss']['val'])
train_acc = np.array(result['acc']['train'])
val_acc = np.array(result['acc']['val'])

start_point=0
end_point=-1

plt.figure(1)
plt.plot(train_loss[start_point:end_point], label='train')
plt.plot(val_loss[start_point:end_point], label='val')
plt.title('%s_%s_%d_loss: %.4f(min val)'%(model, dataset, seed, np.min(val_loss)))
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(train_acc[start_point:end_point], label='train')
plt.plot(val_acc[start_point:end_point], label='val')
plt.title('%s_%s_%d_correlation: %.4f(max val)'%(model, dataset, seed, np.max(val_acc)))
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
