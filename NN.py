from __future__ import print_function, division
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms,utils
import myutils
from image_proprecessing import Rescale, RandomHorizontalFlip, RandomCrop, ToTensor, Normailize, show_landmarks, LandmarksDataset
# Data Augumentation and normalization

'''
# Predictions
def prediction_model(model, test_data):
    was_training = model.training
    model.eval()
    images_so_far = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            # labels = labels.to(device)
            outputs = model(inputs)
            model.train(mode = was_training)
    return  outputs
'''

print("Pytorch CUDA Enabled? ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = 'lfw'
data_transforms = {
    'train':transforms.Compose([
        Rescale(255),
        RandomCrop(227),
        RandomHorizontalFlip(),
        Normailize(),
        ToTensor()
    ]),
    'val': transforms.Compose([
        Rescale(227),
        Normailize(),
        ToTensor(),
    ]),
}

'''
    'test': transforms.Compose([
        Rescale(227),
        ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
'''

transformed_dataset = {x: LandmarksDataset(csv_file = os.path.join(root_dir, x+'.csv'),
                                           root_dir = os.path.join(root_dir, x+'/'),
                                           transform = data_transforms[x])
                       for x in ['train', 'val']}

dataset_sizes = {x: len(transformed_dataset[x]) for x in ['train', 'val']}

# num_workers must be 0 in Windows
dataloaders = {x: DataLoader(transformed_dataset[x],
                             batch_size = 128,
                             shuffle = True,
                             num_workers = 0)
               for x in ['train', 'val']}


def corrections(outputs, labels, radius):
    correct = 0
    for i in range(outputs.size(0)):
        for j in range(outputs.size(1)):
            if (outputs[i, j, 0] - labels[i, j, 0]) ** 2 + (outputs[i, j ,1] - labels[i, j, 1]) ** 2 < radius ** 2:
                correct += 1
    return correct


def train_model(model, dataloaders, criterion, optimizer, radius = 10 , num_epochs = 25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train' :
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            #for i in range(len(dataloaders[phase])):
                #inputs = dataloaders[phase][i]['image'].to(device)
                #landmarks_2d = dataloaders[phase][i]['landmarks_2d'].to(device)
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                inputs = sample_batched['image'].float().to(device)
                landmarks_2d = sample_batched['landmarks_2d'].float().to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.view(outputs.size(0), -1, 2)
                    loss = criterion(outputs, landmarks_2d)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += corrections(outputs, landmarks_2d, radius)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


# Initialize the model
model = torchvision.models.alexnet(pretrained = True)
for param in model.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(9216, 1024)
model.classifier[4] = nn.Linear(1024, 512)
model.classifier[6] = nn.Linear(512, 14)


#model.classifier[1] = nn.Linear(9216, 512)
#model.classifier[4] = nn.Linear(512, 256)
#model.classifier[6] = nn.Linear(256, 14)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
model = train_model(model, dataloaders, criterion, optimizer,  radius = 2 , num_epochs = 70)
torch.save(model, os.path.join(root_dir, 'alexnet.pth'))
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 100, gamma= 0.1)



'''
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs = next(iter(dataloaders['train']))['image']

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out)
plt.show()
'''

'''
transformed_dataset = LandmarksDataset(csv_file= 'lfw/train100.csv', root_dir = 'lfw/train/',
                                       transform = transforms.Compose([Rescale(256), RandomCrop(224),
                                                                       RandomHorizontalFlip(), Normailize(), ToTensor()]))
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks_2d'].size())


dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)


# show transformed pictures in batch

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks_2d']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')
        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks_2d'].size())

    # observe 4th batch and stop.
    if i_batch == 4:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
'''