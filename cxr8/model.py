import torch
import torch.nn as nn
from torchvision import models
from Global_pooling import GlobalPool
from attention import AttentionMap
from efficientnet_pytorch import EfficientNet


# class DenseNet121(nn.Module):
#     """Model modified.
#     The architecture of our model is the same as standard DenseNet121
#     except the classifier layer which has an additional sigmoid function.
#     """
#     def __init__(self, cfg):
#         super(DenseNet121, self).__init__()
#         self.densenet121 = models.densenet121(pretrained=True)
#         num_ftrs = self.densenet121.classifier.in_features
#         self.densenet121.classifier = nn.Sequential(
#             nn.Linear(num_ftrs, cfg.num_classes),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.densenet121(x)
#         return x


class ResNet18(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
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
        x = self.img_model(x)
        return x


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
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
        setattr(self, "attention_map", AttentionMap(self.cfg, self.num_ftrs))

    def forward(self, x):
        # features = [x]
        # for k, v in self.img_model.features._modules.items():
        #     features.append(v(features[-1]))
        # feat_map = features[-1]
        feat_map = self.img_model.features(x)

        if self.cfg.attention_map:
            feat_map = self.attention_map(feat_map)
        x = self.pool(feat_map)
        x = self.drop(x)
        x = self.FF(x)
        x = torch.squeeze(x)
        x = self.sig(x)
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        return x


BACKBONES = {'densenet121': DenseNet121}

BACKBONES_TYPES = {'densenet121': 'densenet',
                   'densenet121_AVG': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet'}


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

    def __init__(self, cfg):
        super(custom_xray, self).__init__()
        self.num_channels = 3
        self.num_outputs = cfg.num_classes if cfg.multi_label else 1
        for x in range(4):
            layer = torch.nn.Sequential(nn.Conv2d(self.num_channels, self.num_channels * 2, kernel_size=3,
                                                  stride=1, padding=1, bias=True),
                                        nn.BatchNorm2d(self.num_channels * 2),
                                        # nn.Dropout(),
                                        # nn.ReLU(inplace=True),
                                        nn.Conv2d(self.num_channels * 2, self.num_channels * 2, kernel_size=3,
                                                  stride=1, padding=1, bias=True),
                                        nn.BatchNorm2d(self.num_channels * 2),
                                        nn.Dropout(),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2, stride=2)
                                        )
            setattr(self, 'layer_' + str(x), layer)
            self.num_channels = self.num_channels * 2

        self.pool = GlobalPool(cfg)
        self.drop = nn.Dropout(0.5)
        self.FF = torch.nn.Sequential(nn.Conv2d(self.num_channels , self.num_outputs, kernel_size=1,
                                                stride=1, padding=0, bias=True))
        self.sig = nn.Sigmoid()

    def forward(self, image):
        for x_ in range(4):
            image = getattr(self, 'layer_' + str(x_))(image)
        image = self.pool(image)
        image = self.drop(image)
        image = self.FF(image).squeeze()
        output = self.sig(image)
        return output
