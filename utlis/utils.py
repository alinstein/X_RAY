import cv2
import numpy as np
import torch
import sys

sys.path.append("..")
from X_RAY.model.model import DenseNet121, ResNet18, EfficientNet_model, custom_xray
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from X_RAY.dataset.dataset import CXRDataset, CXRDatasetBinary
from X_RAY.config import parser

args = parser.parse_args()


def model_name(args):
    """ Returns a name for the model based on CNN, image resolution, pooling layer and number of diseases"""

    if args.attention_map is None:
        return str(
            args.backbone + '_' + str(args.pretrained) + '_' + args.global_pool + '_IMG_SIZE_' + str(args.img_size) +
            "_num_class_" + str(args.num_classes))
    else:
        return str(args.backbone + '_' + str(
            args.pretrained) + '_' + args.global_pool + '_' + args.attention_map + '_IMG_SIZE_'
                   + str(args.img_size) + "_num_class_" + str(args.num_classes))


def select_model(args):
    """Loads and Returns the CNN model"""

    if args.backbone == "densenet121":
        return DenseNet121(args)
    if args.backbone == "ResNet18":
        return ResNet18(args)
    if args.backbone == "EfficientNet":
        return EfficientNet_model(args)
    if args.backbone == "custom":
        return custom_xray(args)


def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):
    """
    Learning rate schedule with respect to epoch
    lr: float, initial learning rate
    lr_factor: float, decreasing factor every epoch_lr
    epoch_now: int, the current epoch
    lr_epochs: list of int, decreasing every epoch in lr_epochs
    return: lr, float, scheduled learning rate.
    """
    count = 0
    for epoch in lr_epochs:
        if epoch_now >= epoch:
            count += 1
            continue
        break
    return lr * np.power(lr_factor, count)


def get_loss(output, target, index, device, cfg):
    target = target[:, index].view(-1)
    if target.sum() == 0:
        loss = torch.tensor(0., requires_grad=True).to(device)
    else:
        weight = (target.size()[0] - target.sum()) / target.sum()
        loss = F.binary_cross_entropy_with_logits(
            output[:, index].view(-1), target.float(), pos_weight=weight)
    label = torch.sigmoid(output[:, index].view(-1)).ge(0.5).float()
    acc = (target == label).float().sum() / len(label)
    return (loss, acc)


def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img
    result = result.div(result.max()).squeeze()
    result = result - result.min()
    result = result / result.max()
    return heatmap, result


def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[2].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.img_model.layer1
        elif layer_num == 2:
            target_layer = arch.img_model.layer2
        elif layer_num == 3:
            target_layer = arch.img_model.layer3
        elif layer_num == 4:
            target_layer = arch.img_model.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 4:
            bottleneck_num = int(hierarchy[3].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 5:
            target_layer = target_layer._modules[hierarchy[4]]

        if len(hierarchy) == 6:
            target_layer = target_layer._modules[hierarchy[5]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if (target_layer_name.split('_')[0] == "img") and (target_layer_name.split('_')[1] == "model"):
        hierarchy = ["img_model"] + target_layer_name.split('_')[2:]
    else:
        hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def make_dataLoader(args):
    """
    Creates train and validation dataloader for multi-label classification.
    The input images are augmented, resized and normalized.

    Parameters
    ----------
    args : configuration file (argparse)

    Returns
    -------
    dataloader: dict containing the train and validation dataloader.
    dataset_sizes: dict containing the size of train dataloader and the size of validation dataloader.
    class_names : the names of the diseases
    """
    trans = {'train': transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), 'val': transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}
    datasets = {'train': CXRDataset(args.data_root_dir, dataset_type='train', Num_classes=args.num_classes,
                                    img_size=args.img_size, transform=trans['train']),
                'val': CXRDataset(args.data_root_dir, dataset_type='val', Num_classes=args.num_classes,
                                  img_size=args.img_size, transform=trans['val'])}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    class_names = datasets['train'].classes
    print("Length of dataset ", dataset_sizes)
    return dataloaders, dataset_sizes, class_names


def make_dataLoader_binary(args):
    """
    Creates train and validation dataloader for binary classification (for abnormal disease).
    The input images are augmented, resized and normalized.

    Parameters
    ----------
    args : configuration file (argparse)

    Returns
    -------
    dataloader: dict containing the train and validation dataloader.
    dataset_sizes: dict containing the size of train dataloader and the size of validation dataloader.
    class_names : the names of the diseases
    """

    trans = {'train': transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), 'val': transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}
    datasets = {'train': CXRDatasetBinary(args.data_root_dir, dataset_type='train',
                                          img_size=args.img_size, transform=trans['train']),
                'val': CXRDatasetBinary(args.data_root_dir, dataset_type='val',
                                        img_size=args.img_size, transform=trans['val'])}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    class_names = ["normal", "abnormal"]

    print("Length of dataset ", dataset_sizes)
    return dataloaders, dataset_sizes, class_names


def LoadModel(checkpoint_file, model, optimizer, epoch_inti, best_auc_ave):
    '''
    The loads the model, optimizer, current epoch, and current validation AUC from the checkpoint location provided.

    Parameters
    ----------
    checkpoint_file: (str) the location of the model in s/m
    model: PyTorch model
    optimizer: PyTorch optimizer
    epoch_inti: current epoch
    best_auc_ave: current best AUC

    Returns
    -------
    Returns the model, optimizer, epoch_inti, best_auc_ave from the saved location.
    '''
    checkpoint = torch.load(checkpoint_file)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_inti = checkpoint['epoch']
    best_auc_ave = checkpoint['best_va_acc']
    if args.num_GPU > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, optimizer, epoch_inti, best_auc_ave


def SaveModel(epoch, model, optimizer, best_auc_ave, file_name):
    """
    Save the model parameters, optimizer, best_AUC

    Parameters
    ----------
    epoch : (int) current epoch
    model : Pytorch model to save
    optimizer : Pytorch optimzer to save
    best_auc_ave : (float) current best AUC
    file_name : (str) location where the model needed to be saved

    """
    if args.num_GPU > 1:
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_va_acc': best_auc_ave
        }
    else:
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_va_acc': best_auc_ave
        }
    torch.save(state, file_name)
    pass


def get_optimizer(params, cfg):
    """
    Loads and returns the optimizer.
    """
    if cfg.optimizer == 'SGD':
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adadelta':
        return Adadelta(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adagrad':
        return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'RMSprop':
        return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(cfg.optimizer))


def weighted_BCELoss(output, target, weights=None):
    '''
    The function computes the weighted Binary cross Entropy loss.
    Parameters
    ----------
    output :  Predicted value from model. (tensor NX8 for multi-label classifcation or Nx1 for binary classification.)
    target : (tensor) Ground truth label. (tensor NX8 for multi-label classifcation or Nx1 for binary classification.)
    weights : the weights (float tensor)

    Returns
    -------
    loss : the WBCE in float tensor
    '''
    output = output.clamp(min=1e-5, max=1 - 1e-5)
    target = target.float()
    if weights is not None:
        assert len(weights) == 2
        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)
    return torch.sum(loss)


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
