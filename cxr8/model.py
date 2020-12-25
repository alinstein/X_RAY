import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from Global_pooling import GlobalPool
from attention import AttentionMap


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, cfg):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs,  cfg.num_classes),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.densenet121(x)
        return x
        
        
class ResNet18_AVG(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(ResNet18_AVG, self).__init__()

        # img model
        img_model = models.resnet18(pretrained=True)
#        num_ftrs = img_model.classifier.in_features
    
        class GlobalAvgPool2d(torch.nn.Module):
    
          def forward(self, x):
            return F.adaptive_avg_pool2d(x, (1, 1))
    
        img_model.avgpool = GlobalAvgPool2d()
        img_model.fc = torch.nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, out_size),
                nn.Sigmoid()
                )
        self.img_model = img_model    

    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
#        features = [x]
#        for k, v in self.img_model.features._modules.items(): features.append( v(features[-1]) )
#        for fea_ve in features:
#          print(k,fea_ve.shape)
          
        x = self.img_model(x)

        return x


class DenseNet121_AVG(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self,cfg):
        super(DenseNet121_AVG, self).__init__()

        # img model
        self.img_model = models.densenet121(pretrained=True)
        self.num_ftrs = self.img_model.classifier.in_features
        self.cfg = cfg
        self.pool = GlobalPool(cfg)
        self.drop = nn.Dropout(0.5)
        self.FF = torch.nn.Sequential(nn.Conv2d(self.num_ftrs, cfg.num_classes, kernel_size=1,
                                                stride=1, padding=0, bias=True))
        self.sig = nn.Sigmoid()
#        self._init_bn()
        self._init_attention_map()
#        
#    def _init_bn(self):
#       for index, num_class in enumerate(self.cfg.num_classes):
#           if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
#               setattr(self, "bn_" + str(index),
#                       nn.BatchNorm2d(512 * self.expand))
#           elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
#               setattr(
#                   self,
#                   "bn_" +
#                   str(index),
#                   nn.BatchNorm2d(
#                       self.backbone.num_features *
#                       self.expand))
#           elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
#               setattr(self, "bn_" + str(index),
#                       nn.BatchNorm2d(2048 * self.expand))
#           else:
#               raise Exception(
#                   'Unknown backbone type : {}'.format(self.cfg.backbone)
#               )
        
    def _init_attention_map(self):
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            setattr(self, "attention_map", AttentionMap(self.cfg, 512))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            setattr(
                self,
                "attention_map",
                AttentionMap(
                    self.cfg,
                    self.num_ftrs))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            setattr(self, "attention_map", AttentionMap(self.cfg, 2048))
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(self.cfg.backbone)
            )

    def forward(self, x):
        features = [x]
        for k, v in self.img_model.features._modules.items(): features.append( v(features[-1]) )
        feat_map = features[-1]
        if self.cfg.attention_map:
            feat_map = self.attention_map(feat_map)
        x = self.pool(feat_map,0)
        x = self.drop(x)
        x = self.FF(x)
        x = torch.squeeze(x)
        x = self.sig(x)
        if len(x.shape) == 1:
            x = torch.unsqueeze(x,0)
        return x


BACKBONES = {'densenet121': DenseNet121_AVG}


BACKBONES_TYPES = {'densenet121': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet'}
                   
                      
class DenseNet121_PCAM(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
        x = self.densenet121(x)
        return x
        
class DenseNet121_ATT(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
        x = self.densenet121(x)
        return x
        
        

