import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import cv2
import random


class CXRDataset(Dataset):

    def __init__(self, root_dir, dataset_type='train', Num_classes=8, img_size=512, transform=None):
        if dataset_type not in ['train', 'val', 'test','box']:
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
        if img_size >= 256:
            self.image_dir = os.path.join(root_dir, "images")
        else :
            self.image_dir = os.path.join(root_dir, "images_resized")
        self.transform = transform
        if self.dataset_type in ['train','val','test']:
            self.index_dir = os.path.join(root_dir, dataset_type + '_label.csv')
            self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0, 1:self.Num_classes + 1].values
            self.label_index = pd.read_csv(self.index_dir, header=0).iloc[:, :self.Num_classes + 1]
        else :
            self.index_dir = os.path.join(root_dir, 'BBox_List_2017.csv')
            #self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0, 1:self.Num_classes + 1].values
            self.label_index = pd.read_csv(self.index_dir, header=0)

            self.index_dir = os.path.join(root_dir, 'label_index.csv')
            self.label_index_test = pd.read_csv(self.index_dir, header=0).iloc[:, :self.Num_classes + 1]


    def __len__(self):
        return int(len(self.label_index))

    def __getitem__(self, idx, ):
        name = self.label_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = cv2.imread(img_dir,0)
        image = np.stack((image,)*3, axis=-1)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        if self.dataset_type in ['train','val','test']:
            label = self.label_index.iloc[idx, 1:self.Num_classes + 1].values.astype('int')
            return image, label, name
        else:
            bbox = self.label_index.iloc[idx, 2:6].values
            label_name = self.label_index.iloc[idx,1]
            label = self.label_index_test.loc[self.label_index_test.iloc[:,0] == name].values[0][1:]
            #.iloc[:, :self.Num_classes + 1].values
            # print(label)
            label = torch.tensor(label.astype(np.float), dtype=torch.float)
            bbox = torch.tensor(bbox.astype(np.float), dtype=torch.float)
            return image, label, bbox, name, label_name

class CXRDatasetBinary(Dataset):

    def __init__(self, root_dir, dataset_type='train', img_size=512, transform=None):
        if dataset_type not in ['train', 'val', 'test1', 'test2']:
            raise ValueError("No such type, must be 'train', 'val', or 'test'")
        self.img_size = img_size

        self.dataset_type = dataset_type
        if img_size >= 256:
            self.image_dir = os.path.join(root_dir, "images")
        else:
            self.image_dir = os.path.join(root_dir, "images_resized")
        self.transform = transform
        if self.dataset_type == 'train':
            self.index_dir = os.path.join(root_dir, 'binary_train.txt')
            self.classes = pd.read_csv(self.index_dir, sep=" ", header=0)
        if self.dataset_type == 'val':
            self.index_dir = os.path.join(root_dir, 'binary_val.txt')
            self.classes = pd.read_csv(self.index_dir, sep=" ", header=0)
        if self.dataset_type == 'test1':
            self.index_dir = os.path.join(root_dir, 'binary_test_attending_rad.txt')
            self.classes = pd.read_csv(self.index_dir, sep=" ", header=0)
        if self.dataset_type == 'test2':
            self.index_dir = os.path.join(root_dir, 'binary_test_rad_consensus_voted.txt')
            self.classes = pd.read_csv(self.index_dir, sep=" ", header=0)
            #
            # self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0,
            #                1:self.Num_classes + 1].values
            # self.label_index = pd.read_csv(self.index_dir, header=0).iloc[:, :self.Num_classes + 1]


    def __len__(self):
        return int(len(self.classes))

    def __getitem__(self, idx, ):
        name = self.classes.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = cv2.imread(img_dir, 0)
        image = np.stack((image,) * 3, axis=-1)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = self.classes.iloc[idx, 1].astype('int')
        return image, label, name



