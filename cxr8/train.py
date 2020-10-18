from dataset import CXRDataset, CXRDataset_BBox_only
from model import Model, DenseNet121 , DenseNet121_AVG , ResNet18_AVG
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import os
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.nn import DataParallel

batch_size =32
num_epochs = 45
learning_rate = 1e-6
output_size = 8
resume_Training = True
regulization = 0
model_save_dir = './savedModels'
model_num=0
model_name = 'net_v1_lr_1e-6_bbox_data_arg'
log_dir = './runs'
data_root_dir = './dataset'

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default='/scratch/alinjose/alinjose/final_project/covid_19/Chexpert/config/', metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('save_path', default='/scratch/alinjose/alinjose/final_project/covid_19/Chexpert/config/', metavar='SAVE_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str,
                    help="GPU indices ""comma separated, e.g. '0,1' ")
parser.add_argument('--pre_train', default=None, type=str, help="If get"
                    "parameters from pretrained model")
parser.add_argument('--resume', default=0, type=int, help="If resume from "
                    "previous run")
parser.add_argument('--logtofile', default=False, type=bool, help="Save log "
                    "in save_path/log.txt if set True")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")

parser.add_argument('--optimizer', default='Adam', type=str, help="Optimizer")
parser.add_argument('--lr', default=1e-6, type=float, help="Learning Rate")
parser.add_argument('--lr_factor', default=0.1, type=float, help="Learning rate factor")
parser.add_argument('--lr_epochs', default=[0.9], type=list, help="Learning rate epoches")
parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
parser.add_argument('--weight_decay', default=0.9, type=float, help="weight decay")

args = parser.parse_args()


mean = [0.50576189]
def make_dataLoader():
    trans = {}
    if model_num == 1 :
            trans['train'] = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor(),
                transforms.Normalize(mean, [1.])
            ])
            trans['val'] = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor(),
                transforms.Normalize(mean, [1.])
            ])
    else :
            trans['train'] = transforms.Compose([
           
                                        transforms.Resize(512),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
            trans['val'] = transforms.Compose([
                                
                                        transforms.Resize(512),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
    
    
    
    datasets = {
        'train': CXRDataset(data_root_dir, transform=trans['train']),
        'val': CXRDataset(data_root_dir, dataset_type='val', transform=trans['val'])
    }
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=6)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    class_names = datasets['train'].classes

    print(dataset_sizes)
    
    return dataloaders, dataset_sizes, class_names




def get_optimizer(params, cfg):
    if cfg.optimizer == 'SGD':
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adadelta':
        return Adadelta(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adagrad':
        return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'RMSprop':
        return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(cfg.optimizer))
        
        
def weighted_BCELoss(output, target, weights=None):
    output = output.clamp(min=1e-5, max=1-1e-5)
    target = target.float()
    if weights is not None:
        assert len(weights) == 2

        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)


    if target.sum() == 0:
        loss_B = torch.tensor(0., requires_grad=True).to(device)
    else:
        weight = (target.size()[0] - target.sum()) / target.sum()
       
        loss_B = F.binary_cross_entropy_with_logits(
                            output.view(-1), target, pos_weight=weight)
    print("_____for loss verificaiton_________")  
    print("Loss with custom method : {:4f} and BCE : {:4f} ",.format(loss,loss_B))

    label = output.view(-1).ge(0.5).float()
    acc = (target == label).float().sum() / len(label)
        
    return torch.sum(loss) , acc

def training(model):
    writer = {x: SummaryWriter(log_dir=os.path.join(log_dir, model_name, x),
                comment=model_name)
          for x in ['train', 'val']}
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0)
    optimizer = get_optimizer(params, cfg)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    dataloaders, dataset_sizes, class_names = make_dataLoader()
    
    
    
    
    
    if not os.path.exists(model_save_dir):
                       os.makedirs(model_save_dir)
    best_auc_ave = 0.0   # to check if best validation accuracy  
    epoch_inti = 1   # epoch starts from here  
    
    best_model_wts = model.state_dict()
    best_auc = []
    iter_num = 0

    # Prepare checkpoint file and model file to save and load from  
    checkpoint_file = os.path.join(model_save_dir, "checkpoint.pth")
    bestmodel_file = os.path.join(model_save_dir, "best_model.pth")      

    ''' Check for existing training results. If it existst, and the configuration
    is set to resume `config.resume_TIRG==True`, resume from previous training. 
    If not, delete existing checkpoint.'''
    if os.path.exists(checkpoint_file):
            if resume_Training:
                print("Checkpoint found! Resuming")
                checkpoint = torch.load(checkpoint_file)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch_inti = checkpoint['epoch']
                best_auc_ave = checkpoint['best_va_acc']
            else:
                pass   

    print("best_auc_ave",best_auc_ave)

    
    since = time.time()

    for epoch in range(epoch_inti,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #scheduler.step()
        # Each epoch has a training and validation phase
        
        for phase in [ 'train','val']:
            
            # Only Validation in every 5 cycles
            if (phase == 'val') and (epoch % 1 != 0):
                continue
                
                
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0
            output_list = []
            label_list = []
            
            # Iterate over data.
            for idx, data in enumerate((dataloaders[phase])):
                # get the inputs
                images, labels, names, bboxes, bbox_valids = data

                images = images.to(device)
                labels = labels.to(device)
                
                if phase == 'train':
                    torch.set_grad_enabled(True)
                else:
                    torch.set_grad_enabled(False)
                    
                #calculate weight for loss
                P = 0
                N = 0
                for label in labels:
                    for v in label:
                        if int(v) == 1: P += 1
                        else: N += 1
                if P!=0 and N!=0:
                    BP = (P + N)/P
                    BN = (P + N)/N
                    weights = torch.tensor([BP, BN], dtype=torch.float).to(device)
                else: weights = None

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(images)
#                # forward
#                outputs, segs = model(images)
                
#                # remove invalid bbox and segmentation outputs
#                bbox_list = []
#                for i in range(bbox_valids.size(0)):
#                    bbox_list.append([])
#                    for j in range(8):
#                        if bbox_valids[i][j] == 1:
#                            bbox_list[i].append(bboxes[i][j])
#                    bbox_list[i] = torch.stack(bbox_list[i]).to(device)
#                
#                seg_list = []
#                for i in range(bbox_valids.size(0)):
#                    seg_list.append([])
#                    for j in range(8):
#                        if bbox_valids[i][j] == 1:
#                            seg_list[i].append(segs[i][j])
#                    seg_list[i] = torch.stack(seg_list[i]).to(device)
                
                # classification loss
                loss , acc = weighted_BCELoss(outputs, labels, weights=weights)
                # segmentation loss
#                for i in range(len(seg_list)):
#                    loss += 5*weighted_BCELoss(seg_list[i], bbox_list[i], weights=torch.tensor([10., 1.]).to(device))/(512*512)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    iter_num += 1

                # metrix
                running_loss += loss.item()
                running_acc += acc
                outputs = outputs.detach().to('cpu').numpy()
                labels =  labels.detach().to('cpu').numpy()
                for i in range(outputs.shape[0]):
                    output_list.append(outputs[i].tolist())
                    label_list.append(labels[i].tolist())
                    
                if idx%10 == 0:
                    if phase == 'train':
                        writer[phase].add_scalar('loss', loss.item()/outputs.shape[0], iter_num)
#                    print('\r{} {:.2f}%'.format(phase, 100*idx/len(dataloaders[phase])), end='\r')
                if idx%100 == 0 and idx!=0:
                    if phase == 'train':
                        try:
                            auc = roc_auc_score(np.array(label_list[-100*batch_size:]), np.array(output_list[-100*batch_size:]))
                            writer[phase].add_scalar('auc', auc, iter_num)
                        except:
                            pass

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_acc / dataset_sizes[phase]
            
            try:
                epoch_auc_ave = roc_auc_score(np.array(label_list), np.array(output_list))
                epoch_auc = roc_auc_score(np.array(label_list), np.array(output_list), average=None)
            except:
                epoch_auc_ave = 0
                epoch_auc = [0 for _ in range(len(class_names))]

            if phase == 'val':
                writer[phase].add_scalar('loss', epoch_loss, iter_num)
                writer[phase].add_scalar('auc', epoch_auc_ave, iter_num)
                writer[phase].add_scalar('acc', epoch_acc, iter_num)
                
            for i, c in enumerate(class_names):
                writer[phase].add_pr_curve(c, np.array(label_list[:][i]), np.array(output_list[:][i]), iter_num)
            
            log_str = ''
            log_str += '{} Loss: {:.4f} AUC: {:.4f} Acc: {:4f} \n\n'.format(
                phase, epoch_loss, epoch_auc_ave, epoch_auc,epoch_acc)
            for i, c in enumerate(class_names):
                log_str += '{}: {:.4f}  \n'.format(c, epoch_auc[i])
            log_str += '\n'
            print(log_str)
            writer[phase].add_text('log',log_str , iter_num)
            print("best_auc_ave",best_auc_ave)
            print("epoch_auc_ave",epoch_auc_ave)
            print("epoch_auc",epoch_auc)
            print("phase",phase)
            print("Acc", epoch_acc)
            # save model for validation
            if phase == 'val' and epoch_auc_ave > best_auc_ave:
                best_auc = epoch_auc
                print("Rewriting",best_auc_ave, "by", epoch_auc_ave)
                best_auc_ave = epoch_auc_ave
                print('Model saved to %s'%(bestmodel_file))
                print("Saving the best checkpoint")

                state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_va_acc': best_auc_ave,
                        }
                torch.save(state, bestmodel_file)
                writer[phase].add_text('log','Model saved to %s\n\n'%(model_save_dir) , iter_num)
                
            if phase == 'train' and epoch % 5 == 0:  
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_va_acc':best_auc_ave,
                    }, checkpoint_file)  

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AUC: {:4f}'.format(best_auc_ave))
    print()
    for i, c in enumerate(class_names):
        print('{}: {:.4f} '.format(c, best_auc[i]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

if __name__ == '__main__':


    model = DenseNet121_AVG(output_size).to(device)
#    model = ResNet18_AVG(output_size).to(device)

#    if args.verbose is True:
#        from torchsummary import summary
#        if cfg.fix_ratio:
#            h, w = cfg.long_side, cfg.long_side
#        else:
#            h, w = cfg.height, cfg.width
#        summary(model.to(device), (3, h, w))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
      
    model.to(device)
    training(model)
