import torch
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import os

def show_landmarks(image, landmarks, ground):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=300, marker='.', c='red', label = 'predicted')
    plt.scatter(ground[:, 0], ground[:, 1], s=300, marker='.', c='blue', label = 'ground choose')
    #plt.pause(0.001)  # pause a bit so that plots are updated
    plt.legend()
    plt.show()

from image_proprecessing import LandmarksDataset, Rescale, Normailize, ToTensor

print("Pytorch CUDA Enabled? ", torch.cuda.is_available())
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

root_dir = 'lfw'
n_radius = 40

test_model = torch.load(os.path.join(root_dir, 'alexnet_jason/alexnet_1024_521_noscheduler.pth'), map_location='cpu')
test_model.to(device)
test_model.eval()

print(test_model)

def corrections(outputs, labels, radius):
    correct = 0
    correct_landmarks = [0] * 7
    print(labels.shape)
    print(outputs.shape)
    for i in range(outputs.size(0)):
        for j in range(outputs.size(1)):
            if (outputs[i, j, 0] - labels[i, j, 0]) ** 2 + (outputs[i, j ,1] - labels[i, j, 1]) ** 2 < radius ** 2:
                correct += 1
                correct_landmarks[j] += 1
    return correct, correct_landmarks

data_transforms = {
    'train':transforms.Compose([
        Rescale(227),
        #RandomCrop(227),
        #RandomHorizontalFlip(),
        Normailize(),
        ToTensor()
    ]),
    'val': transforms.Compose([
        Rescale(227),
        Normailize(),
        ToTensor(),
    ]),
}

test_dataset = LandmarksDataset(csv_file = os.path.join(root_dir, 'test.csv'),
                                root_dir = os.path.join(root_dir, 'test/'),
                                transform = transforms.Compose([
                                    Rescale(227),
                                    Normailize(),
                                    ToTensor(),]))

dataloaders = DataLoader(test_dataset, batch_size = 128, shuffle = True, num_workers = 0)

corrections_list = []
corrections_landmark_list = []
for i in range(n_radius):
    temp_list = [0] * 7
    corrections_list.append(0)
    corrections_landmark_list.append(temp_list)

for i_batch, sample_batched in enumerate(dataloaders):
    landmarks_2d = sample_batched['landmarks_2d'].float().to(device)
    outputs = test_model(sample_batched['image'].float().to(device))
    outputs = outputs.view(outputs.size(0), -1, 2)
    #sample_batched_np = sample_batched['image'][0,:,:,:].numpy().transpose((1, 2, 0))
    #outputs_np = outputs[0,:,:].cpu().data.numpy()
    #show_landmarks(sample_batched_np, outputs_np, landmarks_2d[0,:,:].cpu().data.numpy())
    for radius in range(n_radius):
        temp_corrections, temp_corrections_landmarks = corrections(outputs, landmarks_2d, radius)
        corrections_list[radius] += temp_corrections
        for i in range(7):
            corrections_landmark_list[radius][i] += temp_corrections_landmarks[i]
    print("processing batch #", i_batch)

acc = []
for this_correction in corrections_list:
    acc.append(float(this_correction) / len(test_dataset))

canthus_rr, canthus_rl, canthus_lr, canthus_ll, mouth_corner_r, mouth_corner_l, nose = [], [], [], [], [], [], []

for i in range(n_radius):
    for j in range(7):
        corrections_landmark_list[i][j] /= len(test_dataset)
        if j == 0:
            canthus_rr.append(corrections_landmark_list[i][j])
        elif j == 1:
            canthus_rl.append(corrections_landmark_list[i][j])
        elif j == 2:
            canthus_lr.append(corrections_landmark_list[i][j])
        elif j == 3:
            canthus_ll.append(corrections_landmark_list[i][j])
        elif j == 4:
            mouth_corner_r.append(corrections_landmark_list[i][j])
        elif j == 5:
            mouth_corner_l.append(corrections_landmark_list[i][j])
        elif j == 6:
            nose.append(corrections_landmark_list[i][j])

n_radius_list = []
for i in range(n_radius):
    n_radius_list.append(i/227)

plt.plot(n_radius_list, canthus_rr, label = 'canthus_rr')
plt.plot(n_radius_list, canthus_rl, label = 'canthus_rl')
plt.plot(n_radius_list, canthus_lr, label = 'canthus_lr')
plt.plot(n_radius_list, canthus_ll, label = 'canthus_ll')
plt.plot(n_radius_list, mouth_corner_r, label = 'mouth_corner_r')
plt.plot(n_radius_list, mouth_corner_l, label = 'mouth_corner_l')
plt.plot(n_radius_list, nose, label = 'nose')

plt.legend()
plt.show()

print(acc)