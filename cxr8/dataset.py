import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import pandas as pd
import numpy as np
from PIL import Image
import os
import cv2
import random


class CXRDataset(Dataset):

    def __init__(self, root_dir, dataset_type='train', Num_classes=8, img_size=512, transform=None):
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError("No such type, must be 'train', 'val', or 'test'")
        self.img_size = img_size
        self.Num_classes = Num_classes
        self.disease_categories = {
            'Atelectasis': 0,
            'Cardiomegaly': 1,
            'Effusion': 2,
            'Infiltrate': 3,
            'Mass': 4,
            'Nodule': 5,
            'Pneumonia': 6,
            'Pneumothorax': 7,
            'Consolidation': 8,
            'Edema': 9,
            'Emphysema': 10,
            'Fibrosis': 11,
            'Pleural_Thickening': 12,
            'Hernia': 13}

        # for dis, dis_val in self.disease_categories.items():
        #     if dis_val>self.Num_classes:
        #         del self.disease_categories[dis]

        self.dataset_type = dataset_type
        self.image_dir = os.path.join(root_dir, "images_resized")
        self.transform = transform
        self.index_dir = os.path.join(root_dir, dataset_type + '_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0, 1:self.Num_classes + 1].values
        self.label_index = pd.read_csv(self.index_dir, header=0).iloc[:, :self.Num_classes + 1]

    def __len__(self):
        return int(len(self.label_index))

    def __getitem__(self, idx):
        name = self.label_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = cv2.imread(img_dir,0)
        image = np.stack((image,)*3, axis=-1)
        try :
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)

            label = self.label_index.iloc[idx, 1:self.Num_classes + 1].values.astype('int')
            return image, label, name
        except:
            print("Error image name",img_dir)
