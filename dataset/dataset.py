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
        if dataset_type not in ['train', 'val', 'test', 'box']:
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
        else:
            self.image_dir = os.path.join(root_dir, "images_resized")
        self.transform = transform
        if self.dataset_type in ['train', 'val', 'test']:
            self.index_dir = os.path.join(root_dir, dataset_type + '_label.csv')
            self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0, 1:self.Num_classes + 1].values
            self.label_index = pd.read_csv(self.index_dir, header=0).iloc[:, :self.Num_classes + 1]
        else:
            self.index_dir = os.path.join(root_dir, 'BBox_List_2017.csv')
            # self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).iloc[0, 1:self.Num_classes + 1].values
            self.label_index = pd.read_csv(self.index_dir, header=0)

            self.index_dir = os.path.join(root_dir, 'label_index.csv')
            self.label_index_test = pd.read_csv(self.index_dir, header=0).iloc[:, :self.Num_classes + 1]

    def __len__(self):
        return int(len(self.label_index))

    def __getitem__(self, idx, ):
        name = self.label_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = cv2.imread(img_dir, 0)
        image = np.stack((image,) * 3, axis=-1)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        if self.dataset_type in ['train', 'val', 'test']:
            label = self.label_index.iloc[idx, 1:self.Num_classes + 1].values.astype('int')
            return image, label, name
        else:
            bbox = self.label_index.iloc[idx, 2:6].values
            label_name = self.label_index.iloc[idx, 1]
            label = self.label_index_test.loc[self.label_index_test.iloc[:, 0] == name].values[0][1:]
            # .iloc[:, :self.Num_classes + 1].values
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
            self.index_dir = os.path.join(root_dir, 'train_label.csv')  # binary_train
            self.classes = pd.read_csv(self.index_dir, header=0)  # sep=" ",
        if self.dataset_type == 'val':
            self.index_dir = os.path.join(root_dir, 'val_label.csv')  # binary_val
            self.classes = pd.read_csv(self.index_dir, header=0)  # sep=" ",
        if self.dataset_type == 'test1':
            self.index_dir = os.path.join(root_dir, 'test_label.csv')  # binary_test_attending_rad
            self.classes = pd.read_csv(self.index_dir, header=0)  # sep=" ",
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
        label = self.classes.iloc[idx, 6].astype('int')
        # while 1:
        #     idx = random.randrange(0, self.__len__())
        #
        #     name = self.classes.iloc[idx, 0]
        #     label = self.classes.iloc[idx, 6].astype('int')
        #     if label == 1:
        #         break
        #     else:
        #         if random.random() < 0.3:
        #             break

        img_dir = os.path.join(self.image_dir, name)
        image = cv2.imread(img_dir, 0)
        image = np.stack((image,) * 3, axis=-1)
        try:
            image = Image.fromarray(image)
        except:
            print(img_dir)
        if self.transform:
            image = self.transform(image)

        return image, label, name


class ImageDataset(Dataset):
    def __init__(self, label_path,transform, Num_classes=14, dataset_type='train'):

        self._label_header = None
        self._image_paths = []
        self.transform = transform
        self.Num_classes = Num_classes
        self._labels = []
        self.dataset_type = dataset_type
        dataset_location = label_path + '\\chexpert\\'
        label_path = label_path+'\\chexpert\\CheXpert-v1.0-small\\' + dataset_type + '.csv'
        # self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
        #              {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '-1'}]
        self.classes = pd.read_csv(label_path, header=None, nrows=1).iloc[0, 5:self.Num_classes + 1].values
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = dataset_location + fields[0]
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1':
                                # self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                        if self.dict[0].get(value) == '1' :
                                #self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    # else:
                    #     labels.append(self.dict[2][value])

                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path
                self._labels.append(labels)
                if flg_enhance and self.dataset_type == 'train':
                    for i in range(1):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = np.stack((image,) * 3, axis=-1)
        image = Image.fromarray(image)
        image = self.transform(image)
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self.dataset_type == 'train' or self.dataset_type == 'valid':
            return (image, labels, self._image_paths[idx])
        elif self.dataset_type == 'test':
            return (image, path, self._image_paths[idx])
        elif self.dataset_type == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self.dataset_type))
