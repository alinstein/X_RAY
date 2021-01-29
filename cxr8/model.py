import torch
import torch.nn as nn
import sys
from torchvision import models
from Global_pooling import GlobalPool
from attention import AttentionMap
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


class PCAM_Model(nn.Module):
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
        self.drop = nn.Dropout(0.5)
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
            # if self.cfg.fc_bn:
            #     bn = getattr(self, "bn_" + str(index))
            #     feat = bn(feat)
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


class WildcatPool2dFunction(Function):
    def __init__(self, kmax, kmin, alpha):
        super(WildcatPool2dFunction, self).__init__()
        WildcatPool2dFunction.kmax = kmax
        WildcatPool2dFunction.kmin = kmin
        WildcatPool2dFunction.alpha = alpha

    @staticmethod
    def forward(ctx, input):
        batch_size = input.size(0)
        num_channels = input.size(1)
        ctx.kmax = WildcatPool2dFunction.kmax
        ctx.kmin = WildcatPool2dFunction.kmin
        ctx.alpha = WildcatPool2dFunction.alpha
        h = input.size(2)
        w = input.size(3)
        n = h * w  # number of regions

        def get_positive_k(k, n):
            if k <= 0:
                return 0
            elif k < 1:
                return round(k * n)
            elif k > n:
                return int(n)
            else:
                return int(k)

        kmax = get_positive_k(ctx.kmax, n)
        kmin = get_positive_k(ctx.kmin, n)
        sorted, indices = input.new(), input.new().long()
        torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True, out=(sorted, indices))
        ctx.indices_max = indices.narrow(2, 0, kmax)
        output = sorted.narrow(2, 0, kmax).sum(2).div_(kmax)

        if kmin > 0 and ctx.alpha is not 0:
            ctx.indices_min = indices.narrow(2, n - kmin, kmin)
            output.add_(sorted.narrow(2, n - kmin, kmin).sum(2).mul_(ctx.alpha / kmin)).div_(2)
        ctx.save_for_backward(input)
        return output.view(batch_size, num_channels)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)
        n = h * w  # number of regions

        def get_positive_k(k, n):
            if k <= 0:
                return 0
            elif k < 1:
                return round(k * n)
            elif k > n:
                return int(n)
            else:
                return int(k)

        kmax = get_positive_k(ctx.kmax, n)
        kmin = get_positive_k(ctx.kmin, n)
        grad_output_max = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmax)
        grad_input = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, ctx.indices_max,
                                                                                              grad_output_max).div_(
            kmax)
        if kmin > 0 and ctx.alpha is not 0:
            grad_output_min = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmin)
            grad_input_min = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2,
                                                                                                      ctx.indices_min,
                                                                                                      grad_output_min).mul_(
                ctx.alpha / kmin)
            grad_input.add_(grad_input_min).div_(2)
        return grad_input.view(batch_size, num_channels, h, w)


class WildcatPool2d(nn.Module):
    def __init__(self, kmax=1, kmin=None, alpha=1):
        super(WildcatPool2d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax
        self.alpha = alpha

    def forward(self, input):
        return WildcatPool2dFunction(self.kmax, self.kmin, self.alpha).apply(input)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ', kmin=' + str(self.kmin) + ', alpha=' + str(
            self.alpha) + ')'


class ClassWisePoolFunction(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        # batch dimension
        batch_size, num_channels, h, w = input.size()
        self.num_maps = self.num_maps
        if num_channels % self.num_maps != 0:
            print(
                'Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)
        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        # ctx.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps


class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction(self.num_maps)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' (num_maps={num_maps})'.format(num_maps=self.num_maps)


class ResNetWSL(nn.Module):
    def __init__(self, model, num_classes, pooling=WildcatPool2d(), dense=False):
        super(ResNetWSL, self).__init__()

        self.dense = dense

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.sig = nn.Sigmoid()
        self.spatial_pooling = pooling

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)
        x = self.sig(x)
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]


def resnet50_wildcat(cfg, kmax=1, kmin=None, alpha=0.7, num_maps=24):
    model = models.resnet18(pretrained=cfg.pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))  # ClassWisePool(num_maps)
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))  #
    return ResNetWSL(model, cfg.num_classes * num_maps, pooling=pooling)

# def resnet101_wildcat(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
#     model = models.resnet101(pretrained)
#     pooling = nn.Sequential()
#     pooling.add_module('class_wise', ClassWisePool(num_maps))
#     pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
#     return ResNetWSL(model, num_classes * num_maps, pooling=pooling)
