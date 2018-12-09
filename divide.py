import os
import shutil
import pandas as pd
import random


root_dir = "E:\\face_cropped_collect"
train_dir = "E:\\face_cropped_divide\\train"
val_dir = "E:\\face_cropped_divide\\val"

train_landmarks = []
val_landmarks = []
landmarks_frame = pd.read_csv("E:\\face_cropped\\train_cropped.csv", header = None)
img_name = landmarks_frame.iloc[:, 0]
landmarks = landmarks_frame.iloc[:, 1:].values
for i in range(len(img_name)):
    if random.randint(1, 100) <= 25:
        file_path = os.path.join(root_dir, img_name[i])
        shutil.copy(file_path, val_dir)
        temp = []
        temp.append(img_name[i])
        for l in landmarks[i]:
            temp.append(l)
        val_landmarks.append(temp)
    else:
        file_path = os.path.join(root_dir, img_name[i])
        shutil.copy(file_path, train_dir)
        temp = []
        temp.append(img_name[i])
        for l in landmarks[i]:
            temp.append(l)
        train_landmarks.append(temp)
    print("processing picture #" + str(i))
pd.DataFrame.from_dict(train_landmarks).to_csv('E:\\face_cropped_divide\\train.csv')
pd.DataFrame.from_dict(val_landmarks).to_csv('E:\\face_cropped_divide\\val.csv')

