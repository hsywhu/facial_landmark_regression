import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

train_loss_path = 'lfw\\loss\\epoch_loss_train.csv'
val_loss_path = 'lfw\\loss\\epoch_loss_val.csv'

train_loss_frame = pd.read_csv(train_loss_path, header = None)
val_loss_frame = pd.read_csv(val_loss_path, header = None)

train_loss = np.asarray(train_loss_frame.iloc[:, 0])
val_loss = np.asarray(val_loss_frame.iloc[:, 0])

x = []
for i in range(len(train_loss_frame)):
    x.append(i)

plt.plot(x, train_loss, color = 'blue', label = 'train_loss')
plt.plot(x, val_loss, color = 'orange', label = 'val_loss')

plt.legend()
plt.show()