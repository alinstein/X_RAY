from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from torch.utils.data.dataset import random_split
import re
import tqdm
from dataset import Covid_Dataset




def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
    
def create_model(device):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    return exp_lr_scheduler,criterion,model_ft
  
  
def train_func(data):
     optimizer.zero_grad()
     text, label = data['review'], data['Sentiment']
     output = model(text)
     loss = criterion(output, label)
     train_loss = loss.item()
     loss.backward()
     optimizer.step()
     train_acc = (output.argmax(1) == label).sum().item()
     return train_loss , train_acc

def test_func(data):
     text, label = data['review'], data['Sentiment']
     output = model(text)
     loss = criterion(output, label)
     test_loss = loss.item()
     test_acc = (output.argmax(1) == label).sum().item()
     return test_loss , test_acc
                


#def get_transform(train):
#    transforms = []
#    transforms.append(T.ToTensor())
#    if train:
#        transforms.append(T.RandomHorizontalFlip(0.5))
#    return T.Compose(transforms)
    
                        
def main():

        N_EPOCHS = 25
        BATCH_SIZE=100
        
        
        
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        dataset_train = Covid_Dataset(csv_file='./Data_Entry_2017.csv', split='train',transform=True)
        #dataset_train = Covid_Dataset(csv_file='./Data_Entry_2017.csv', split='train',transform=True)
        criterion = torch.nn.CrossEntropyLoss().to(device)
#        sub_train_ = Sentiment_Dataset()
#        sub_test_ = Sentiment_Dataset(split="test")
#        vocab_size=len(sub_train_.unique())
        dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                        shuffle=True)

#        dataloader_test = DataLoader(sub_test_, batch_size=BATCH_SIZE)
       
        print("Building Model")
        exp_lr_scheduler,criterion,model_ft = model=create_model(device)
  
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


        return 0



if __name__ == '__main__':
  main()
