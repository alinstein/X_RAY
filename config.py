import argparse
import torch
import os

num_GPU = torch.cuda.device_count()
current_location = os.getcwd()
data_root_dir = os.path.join(current_location, 'dataset')

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--num_classes', default=8, type=int, help="Number of disease labels")
parser.add_argument('--num_workers', default=3, type=int, help="Number of workers for each data loader")
parser.add_argument('--batch_size', default=32, type=int, help="Batch size while training")
parser.add_argument('--device_ids', default='0', type=str, help="GPU indices ""comma separated, e.g. '0,1' ")

parser.add_argument('--multi_label', default=True, type=bool, help="False if model to run is abnormal X-rays")
parser.add_argument('--resume', default=False, type=bool, help="If resume from previous run")
parser.add_argument('--model_save_dir', default='./savedModels', type=str, help="Location of saved models")
parser.add_argument('--log_dir', default='./runs', type=str, help="Location to save logs")
parser.add_argument('--global_pool', default='LSE', type=str, help="Global Pooling method [LSE,AVG,MAX,PCAM]")
parser.add_argument('--backbone', default='ResNet18', type=str,
                    help="Backbone Network [densenet121, ResNet18, EfficientNet, custom]")
parser.add_argument('--pretrained', default=True, type=bool, help="Use the pretrained weights of backbone")
parser.add_argument('--img_size', default=224, type=int, help="Resolution of input image")
parser.add_argument('--attention_map', default=None, type=str, help="attention")
parser.add_argument('--lse_gamma', default=10, type=float, help="lse_gamma")

parser.add_argument('--logtofile', default=False, type=bool, help="Save log in save_path/log.txt if set True")
parser.add_argument('--optimizer', default='Adam', type=str, help="Optimizer")
parser.add_argument('--epochs', default=1, type=int, help="Number of epochs")
parser.add_argument('--lr', default=1e-5, type=float, help="Learning Rate")
parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
parser.add_argument('--weight_decay', default=0.0, type=float, help="weight decay")
parser.add_argument('--data_root_dir', default=data_root_dir, type=str, help="Current location")
parser.add_argument('--num_GPU', default=num_GPU, type=int, help="Number of GPU")

