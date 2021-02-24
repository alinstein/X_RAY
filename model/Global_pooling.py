import torch
from torch import nn

# Referenced from https://github.com/jfhealthcare/Chexpert
class PcamPool(nn.Module):

    def __init__(self):
        super(PcamPool, self).__init__()

    def forward(self, feat_map, logit_map):
        assert logit_map is not None

        prob_map = torch.sigmoid(logit_map)
        weight_map = prob_map / prob_map.sum(dim=2, keepdim=True) \
            .sum(dim=3, keepdim=True)
        feat = (feat_map * weight_map).sum(dim=2, keepdim=True) \
            .sum(dim=3, keepdim=True)

        return feat


class LogSumExpPool(nn.Module):

    def __init__(self, gamma):
        super(LogSumExpPool, self).__init__()
        self.gamma = gamma

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        (N, C, H, W) = feat_map.shape

        # (N, C, 1, 1) m
        m, _ = torch.max(
            feat_map, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)

        # (N, C, H, W) value0
        value0 = feat_map - m
        area = 1.0 / (H * W)
        g = self.gamma

        # TODO: split dim=(-1, -2) for onnx.export
        return m + 1 / g * torch.log(area * torch.sum(
            torch.exp(g * value0), dim=(-1, -2), keepdim=True))

class GlobalPool(nn.Module):
    def __init__(self, cfg):
        super(GlobalPool, self).__init__()
        self.cfg = cfg
        if self.cfg.global_pool == 'AVG':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.cfg.global_pool == 'MAX':
            self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        elif self.cfg.global_pool == 'PCAM':
            self.pcampool = PcamPool()
        elif self.cfg.global_pool == 'LSE':
            self.lse_pool = LogSumExpPool(cfg.lse_gamma)

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, feat_map, logit_map=None):
        if self.cfg.global_pool == 'AVG':
            return self.avgpool(feat_map)
        elif self.cfg.global_pool == 'MAX':
            return self.maxpool(feat_map)
        elif self.cfg.global_pool == 'PCAM':
            return self.pcampool(feat_map, logit_map)
        elif self.cfg.global_pool == 'EXP':
            return self.exp_pool(feat_map)
        elif self.cfg.global_pool == 'LINEAR':
            return self.linear_pool(feat_map)
        elif self.cfg.global_pool == 'LSE':
            return self.lse_pool(feat_map)
        else:
            raise Exception('Unknown pooling type : {}'
                            .format(self.cfg.global_pool))
