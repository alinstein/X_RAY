import torch
import torch.nn as nn
import sys
from torchvision import models
from .Global_pooling import GlobalPool
from .attention import AttentionMap
from efficientnet_pytorch import EfficientNet
from torch.autograd import Function
import torch.nn.functional as F


class ResNet18(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard ResNet18
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, cfg):
        super(ResNet18, self).__init__()
        img_model = models.resnet18(pretrained=cfg.pretrained)
        self.cfg = cfg
        self.num_outputs = cfg.num_classes if cfg.multi_label else 1
        img_model.avgpool = GlobalPool(cfg)
        img_model.fc = torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_outputs),
            nn.Sigmoid()
        )
        self.img_model = img_model

    def forward(self, x):
        """
        :param x: input image [size N X H X W X C]
        :return x: probability of each disease [size N X 8]
        """
        x = self.img_model(x)
        return x


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
    """

    def __init__(self, cfg):
        super(DenseNet121, self).__init__()
        self.img_model = models.densenet121(pretrained=cfg.pretrained)
        self.num_ftrs = self.img_model.classifier.in_features
        self.num_outputs = cfg.num_classes if cfg.multi_label else 1
        self.cfg = cfg
        self.pool = GlobalPool(cfg)
        self.drop = nn.Dropout(0.5)
        self.FF = torch.nn.Sequential(nn.Conv2d(self.num_ftrs, self.num_outputs, kernel_size=1,
                                                stride=1, padding=0, bias=True))
        self.sig = nn.Sigmoid()
        if cfg.attention_map:
            self._init_attention_map()

    def _init_attention_map(self):
        setattr(self, "attention_map_1", AttentionMap(self.cfg, self.num_ftrs))


    def forward(self, x):
        """
        :param x: input image [size N X H X W X C]
        :return x: probability of each disease [size N X 8]
        """
        feat_map = x
        for k, v in self.img_model.features._modules.items():
            feat_map = v(feat_map)
            if self.cfg.attention_map:
                feat_map = self.attention_map(feat_map)
        feat_map = self.img_model.features(x)
        x = self.pool(feat_map)
        x = self.drop(x)
        x = self.FF(x)
        x = self.sig(x)
        if len(x.shape) > 2:
            x = torch.squeeze(x, -1)
        if len(x.shape) > 2:
            x = torch.squeeze(x, -1)
        return x


class PCAM_Model(nn.Module):
    """Probabilistic Class Activation Map,
    Ref; https://arxiv.org/pdf/2005.14480.pdf
    """
    def __init__(self, cfg):
        super(PCAM_Model, self).__init__()

        self.cfg = cfg
        if self.cfg.backbone == 'densenet121':
            self.img_model = models.densenet121(pretrained=cfg.pretrained)
            self.num_ftrs = self.img_model.classifier.in_features
        elif self.cfg.backbone == 'ResNet18':
            self.img_model = models.resnet18(pretrained=cfg.pretrained)
            self.num_ftrs = 512
            self.img_model = nn.Sequential(
                self.img_model.conv1,
                self.img_model.bn1,
                self.img_model.relu,
                self.img_model.maxpool,
                self.img_model.layer1,
                self.img_model.layer2,
                self.img_model.layer3,
                self.img_model.layer4)

        self.num_outputs = cfg.num_classes
        self.cfg.global_pool = 'PCAM'
        self.global_pool = GlobalPool(cfg)
        self.drop = nn.Dropout(0.0)
        self.sig = nn.Sigmoid()
        if cfg.attention_map:
            self._init_attention_map()
        self._init_classifier()

    def _init_classifier(self):
        for index in range((self.cfg.num_classes)):
            if self.cfg.backbone == 'ResNet18':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        self.num_ftrs,
                        1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif self.cfg.backbone == 'densenet121':
                setattr(
                    self,
                    "fc_" +
                    str(index),
                    nn.Conv2d(
                        self.num_ftrs,
                        1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_attention_map(self):
        setattr(self, "attention_map", AttentionMap(self.cfg, self.num_ftrs))

    def forward(self, x):
        """
        :param x: input image [size N X H X W X C]
        :return: problity of each disease [size N X 8]
        """
        if self.cfg.backbone == 'densenet121':
            feat_map = self.img_model.features(x)
        elif self.cfg.backbone == 'ResNet18':
            feat_map = self.img_model(x)

        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index in range((self.cfg.num_classes)):
            if self.cfg.attention_map:
                feat_map = self.attention_map(feat_map)

            classifier = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            logit_map = classifier(feat_map)
            logit_maps.append(logit_map.squeeze())
            # (N, C, 1, 1)
            feat = self.global_pool(feat_map, logit_map)

            feat = F.dropout(feat)
            # (N, num_class, 1, 1)

            logit = classifier(feat)
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            logits.append(logit)

        logits = torch.cat(logits, dim=1)
        return logits, logit_maps


class EfficientNet_model(nn.Module):

    def __init__(self, cfg):
        super(EfficientNet_model, self).__init__()
        self.img_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.num_ftrs = 1280
        self.cfg = cfg
        self.num_outputs = cfg.num_classes if cfg.multi_label else 1
        self.pool = GlobalPool(cfg)
        self.drop = nn.Dropout(0.5)
        self.FF = torch.nn.Sequential(nn.Conv2d(self.num_ftrs, self.num_outputs, kernel_size=1,
                                                stride=1, padding=0, bias=True))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        :param image: input image [size N X H X W X C]
        :return: problity of each disease [size N X 8]
        """
        feat_map = self.img_model.extract_features(x)
        x = self.pool(feat_map)
        x = self.drop(x)
        x = self.FF(x)
        x = torch.squeeze(x)
        x = self.sig(x)
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        return x


class custom_xray(nn.Module):
    """
    A custom model similar to ResNet architecture but smaller is size
    """
    def __init__(self, cfg):
        """
            Args : cfg :input configurations

        """
        super(custom_xray, self).__init__()
        self.num_channels = 3
        self.num_outputs = cfg.num_classes if cfg.multi_label else 1
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, self.num_channels * 5, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(self.num_channels * 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.num_channels = self.num_channels * 5
        for x in range(4):
            if x == 0:
                coff = 1
            else:
                coff = 1.5

            layer = torch.nn.Sequential(
                nn.BatchNorm2d(int(self.num_channels * coff)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(self.num_channels * coff), self.num_channels * 2, kernel_size=3,
                          stride=1, padding=1, bias=True),

                nn.BatchNorm2d(self.num_channels * 2),

                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_channels * 2, self.num_channels * 2, kernel_size=1,
                          stride=1, padding=0, bias=True),

                nn.BatchNorm2d(self.num_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_channels * 2, self.num_channels * 2, kernel_size=3,
                          stride=1, padding=1, bias=True),

                nn.BatchNorm2d(self.num_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_channels * 2, self.num_channels * 2, kernel_size=1,
                          stride=1, padding=0, bias=True),

                nn.BatchNorm2d(self.num_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_channels * 2, self.num_channels * 2, kernel_size=1,
                          stride=1, padding=0, bias=True),

                nn.BatchNorm2d(self.num_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_channels * 2, self.num_channels * 2, kernel_size=1,
                          stride=1, padding=0, bias=True),

                nn.BatchNorm2d(self.num_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_channels * 2, self.num_channels * 2, kernel_size=1,
                          stride=1, padding=0, bias=True),

                nn.Dropout(),
                nn.AvgPool2d(2, stride=2)
            )
            setattr(self, 'layer_' + str(x), layer)

            skip_layer = torch.nn.Sequential(
                nn.BatchNorm2d(int(self.num_channels * coff)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(self.num_channels * coff), self.num_channels, kernel_size=1,
                          stride=1, padding=0, bias=True),
                nn.AvgPool2d(2, stride=2)
            )
            setattr(self, 'skiplayer_' + str(x), skip_layer)
            self.num_channels = self.num_channels * 2

        self.pool = GlobalPool(cfg)
        self.drop = nn.Dropout(0.5)
        self.FF = torch.nn.Sequential(nn.Conv2d(int(self.num_channels * 1.5), self.num_outputs, kernel_size=1,
                                                stride=1, padding=0, bias=True))
        self.sig = nn.Sigmoid()

    def forward(self, image):
        """
        :param image: input image [size N X H X W X C]
        :return: problity of each disease [size N X 8]
        """
        image = self.first_layer(image)
        for x_ in range(4):
            image_ = getattr(self, 'skiplayer_' + str(x_))(image)
            image = getattr(self, 'layer_' + str(x_))(image)
            image = torch.cat([image, image_], dim=1)
        image = self.pool(image)
        image = self.drop(image)
        image = self.FF(image).squeeze()
        output = self.sig(image)
        return output
