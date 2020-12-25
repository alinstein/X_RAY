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
#                nn.Conv2d(
#                            512,
#                            out_size,
#                            kernel_size=1,
#                            stride=1,
#                            padding=0,
#                            bias=True),
                nn.Sigmoid()
                )
        self.img_model = img_model    

    def forward(self, x):
#        print(x.shape)
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
    def __init__(self, out_size,cfg):
        super(DenseNet121_AVG, self).__init__()

        # img model
        self.img_model = models.densenet121(pretrained=True)
        self.num_ftrs = self.img_model.classifier.in_features
    
        self.pool = GlobalPool(cfg)
        self.cfg = cfg
        self.drop = nn.Dropout(0.5)
#                nn.Linear(512, out_size),
        self.FF = torch.nn.Sequential(
                                    nn.Conv2d(
                                                self.num_ftrs,
                                                out_size,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=True),
                                    
                                    )
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

        x = torch.cat((x, x, x), dim=1)
        features = [x]
        for k, v in self.img_model.features._modules.items(): features.append( v(features[-1]) )
          
        feat_map = features[-1]
        if self.cfg.attention_map != None:
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
#        print(x.shape)
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
#        print(x.shape)
        x = torch.cat((x, x, x), dim=1)
        x = self.densenet121(x)
        return x
        
        
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seg = GCN(14, 512)
        self.features = nn.Sequential(
            #512x512x15
            nn.Conv2d(15, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #256x256
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #128x128
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #64x64
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #32x32
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #16x16
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
            #8x8
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 8),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        seg = torch.sigmoid(self.seg(torch.cat((x, x, x), dim=1)))
        out = self.features(torch.cat((seg, x), dim=1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, seg

class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class GCN(nn.Module):
    def __init__(self, num_classes, input_size):
        super(GCN, self).__init__()
        self.input_size = input_size
        resnet = models.resnet152(pretrained=True)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = _GlobalConvModule(2048, num_classes, (7, 7))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (7, 7))
        self.gcm3 = _GlobalConvModule(512, num_classes, (7, 7))
        self.gcm4 = _GlobalConvModule(256, num_classes, (7, 7))

        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)

    def forward(self, x):
        # if x: 512
        fm0 = self.layer0(x)  # 256
        fm1 = self.layer1(fm0)  # 128
        fm2 = self.layer2(fm1)  # 64
        fm3 = self.layer3(fm2)  # 32
        fm4 = self.layer4(fm3)  # 16

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128

        fs1 = self.brm5(F.interpolate(gcfm1, fm3.size()[2:], mode='bilinear',align_corners=True) + gcfm2)  # 32
        fs2 = self.brm6(F.interpolate(fs1, fm2.size()[2:], mode='bilinear',align_corners=True) + gcfm3)  # 64
        fs3 = self.brm7(F.interpolate(fs2, fm1.size()[2:], mode='bilinear',align_corners=True) + gcfm4)  # 128
        fs4 = self.brm8(F.interpolate(fs3, fm0.size()[2:], mode='bilinear',align_corners=True))  # 256
        out = self.brm9(F.interpolate(fs4, self.input_size, mode='bilinear',align_corners=True))  # 512

        return out
