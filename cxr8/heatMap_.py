import torch
from model import Model, DenseNet121 , DenseNet121_AVG , ResNet18_AVG
from torchvision import transforms
from PIL import Image
import cv2
import os
import matplotlib.pyplot  as plt
import numpy as np
from train import parser

class HeatmapGenerator():

    # ---- Initialize heatmap generator
    # ---- pathModel - path to the trained densenet model
    # ---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    # ---- nnClassCount - class count, 14 for chxray-14

    def __init__(self, pathModel, nnClassCount, transCrop):
        # Class names
        self.class_names = ['Atelectasis','Cardiomegaly',
                            'Effusion', 'Infiltrate','Mass', 'Nodule',
                            'Pneumonia', 'Pneumothorax', 'Consolidation',
                            'Edema','Emphysema','Fibrosis',
                            'Pleural_Thickening','Hernia']

        # ---- Initialize the network
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_GPU = torch.cuda.device_count()
        args = parser.parse_args()
        model = DenseNet121_AVG(nnClassCount,args).cuda()

        if torch.cuda.is_available():
            model = (model).cuda()
        else:
            model = (model)

        modelCheckpoint = torch.load(pathModel)
        if num_GPU > 1:
            model.module.load_state_dict(modelCheckpoint['model_state_dict'])
        else:
            model.load_state_dict(modelCheckpoint['model_state_dict'])


        self.model = model
        self.model.eval()

        # ---- Initialize the weights
        self.weights = list(self.model.img_model.features.parameters())[-2]

        # ---- Initialize the image transform
        normalize = transforms.Normalize([0.5], [0.5])
        transformList = []
        transformList.append(transforms.Resize((transCrop, transCrop)))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transformSequence = transforms.Compose(transformList)

    # --------------------------------------------------------------------------------

    def generate(self, pathImageFile, pathOutputFile, transCrop):

        # ---- Load image, transform, convert
        with torch.no_grad():

            #imageData = Image.open(pathImageFile).convert('RGB')
            imageData = Image.open(pathImageFile).convert('L')
            imageData = self.transformSequence(imageData)
            imageData = imageData.unsqueeze_(0)
            if torch.cuda.is_available():
                imageData = imageData.cuda()
            l = self.model(imageData)
            image_squeeze = torch.cat((imageData,imageData, imageData), dim=1)

            output = self.model.img_model.features(image_squeeze)
            label = self.class_names[torch.max(l, 1)[1]]
            # ---- Generate heatmap
            heatmap = None
            for i in range(0, len(self.weights)):
                map = output[0, i, :, :]
                if i == 0:
                    heatmap = self.weights[i] * map
                else:
                    heatmap += self.weights[i] * map
                npHeatmap = heatmap.cpu().data.numpy()

        # ---- Blend original and heatmap

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))

        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        img = cv2.addWeighted(imgOriginal, 1, heatmap, 0.35, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(label)
        plt.imshow(img)
        plt.plot()
        plt.axis('off')
        plt.savefig(pathOutputFile)
        plt.show()



if __name__ == '__main__':
    current_location = os.getcwd()

    image_loc = "dataset/images/00000003_001.png"
    pathOutput = os.path.join(current_location,"heatmap_output")
    if not os.path.exists(pathOutput):
        os.makedirs(pathOutput)

    image_loc_out = "0000003_001.png"
    pathOutputImage = os.path.join(pathOutput,"heatmap_"+image_loc_out)
    image_loc = os.path.join(current_location,image_loc)

    pathModel = os.path.join(current_location,"savedModels", "densenet121_AVG_best_model.pth")
    nnClassCount = 14
    imgtransCrop = 512

    h = HeatmapGenerator(pathModel, nnClassCount, imgtransCrop)
    h.generate(image_loc, pathOutputImage, imgtransCrop)