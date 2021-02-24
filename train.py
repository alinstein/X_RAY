import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.nn import DataParallel
import torch.nn.functional as F
from tensorboardX import SummaryWriter


from sklearn.metrics import roc_auc_score
import numpy as np
import time
from tqdm import tqdm
import os

from eval import eval_function
from config import parser
from dataset.dataset import CXRDataset, CXRDatasetBinary
from utlis.utils import model_name, select_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_GPU = torch.cuda.device_count()
current_location = os.getcwd()
data_root_dir = os.path.join(current_location, 'dataset')


def make_dataLoader(args):
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
    datasets = {'train': CXRDataset(data_root_dir, dataset_type='train', Num_classes=args.num_classes,
                                    img_size=args.img_size, transform=trans['train']),
                'val': CXRDataset(data_root_dir, dataset_type='val', Num_classes=args.num_classes,
                                  img_size=args.img_size, transform=trans['val'])}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    class_names = datasets['train'].classes
    print("Length of dataset ", dataset_sizes)
    return dataloaders, dataset_sizes, class_names


def make_dataLoader_binary(args):
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
    datasets = {'train': CXRDatasetBinary(data_root_dir, dataset_type='train',
                                          img_size=args.img_size, transform=trans['train']),
                'val': CXRDatasetBinary(data_root_dir, dataset_type='val',
                                        img_size=args.img_size, transform=trans['val'])}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    class_names = ["normal", "abnormal"]

    print("Length of dataset ", dataset_sizes)
    return dataloaders, dataset_sizes, class_names


def LoadModel(checkpoint_file, model, optimizer, epoch_inti, best_auc_ave):
    checkpoint = torch.load(checkpoint_file)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_inti = checkpoint['epoch']
    best_auc_ave = checkpoint['best_va_acc']
    if num_GPU > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, optimizer, epoch_inti, best_auc_ave


def SaveModel(epoch, model, optimizer, best_auc_ave, file_name):
    if num_GPU > 1:
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
    output = output.clamp(min=1e-5, max=1 - 1e-5)
    target = target.float()
    if weights is not None:
        assert len(weights) == 2
        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)
    return torch.sum(loss)


def training(model, args):
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, model_name(args)))
    writer.add_text('log', str(args), 0)
    if args.backbone == 'resnet50_wildcat':
        params = model.get_config_optim(args.lr, 0.5)
    else:
        params = model.parameters()
    optimizer = get_optimizer(params, args)
    dataloaders, dataset_sizes, class_names = make_dataLoader(args)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    best_auc_ave = 0.0  # to check if best validation accuracy
    epoch_inti = 1  # epoch starts from here

    best_model_wts = model.state_dict()
    best_auc = []
    iter_num = 0

    # Prepare checkpoint file and model file to save and load from  
    checkpoint_file = os.path.join(args.model_save_dir, str(model_name(args) + '_' + "checkpoint.pth"))
    bestmodel_file = os.path.join(args.model_save_dir, str(model_name(args) + '_' + "best_model.pth"))

    ''' Check for existing training results. If it existst, and the configuration
    is set to resume `config.resume_TIRG==True`, resume from previous training. 
    If not, delete existing checkpoint.'''
    if os.path.exists(checkpoint_file):
        if args.resume:
            model, optimizer, epoch_inti, best_auc_ave = LoadModel(checkpoint_file, model, optimizer, epoch_inti,
                                                                   best_auc_ave)
            print("Checkpoint found! Resuming")
        else:
            pass

    print(model)
    since = time.time()

    for epoch in range(epoch_inti, args.epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.epochs))
        print('-' * 10)

        for phase in ['train', 'val']:

            # Only Validation in every 5 cycles
            if (phase == 'val') and (epoch % 1 != 0):
                continue
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            output_list = []
            label_list = []

            # Iterate over data.
            for idx, data in enumerate(tqdm(dataloaders[phase])):

                # if idx >= 10:
                #     break
                images, labels, names = data
                images = images.to(device)
                labels = labels.to(device)

                if phase == 'train':
                    torch.set_grad_enabled(True)
                else:
                    torch.set_grad_enabled(False)

                # calculate weight for loss
                P = 0
                N = 0
                for label in labels:
                    for v in label:
                        if int(v) == 1:
                            P += 1
                        else:
                            N += 1
                if P != 0 and N != 0:
                    BP = (P + N) / P
                    BN = (P + N) / N
                    weights = torch.tensor([BP, BN], dtype=torch.float).to(device)
                else:
                    weights = None

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(images)

                # classification loss
                loss = weighted_BCELoss(outputs, labels, weights=weights)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    iter_num += 1

                running_loss += loss.item()
                outputs = outputs.detach().to('cpu').numpy()
                labels = labels.detach().to('cpu').numpy()

                for i in range(outputs.shape[0]):
                    output_list.append(outputs[i].tolist())
                    label_list.append(labels[i].tolist())

                if idx % 100 == 0 and idx != 0:
                    if phase == 'train':
                        writer.add_scalar('loss/train_batch', loss.item() / outputs.shape[0], iter_num)
                        try:
                            auc = roc_auc_score(np.array(label_list[-100 * args.batch_size:]),
                                                np.array(output_list[-100 * args.batch_size:]))
                            writer.add_scalar('auc/train_batch', auc, iter_num)
                        except:
                            pass

            epoch_loss = running_loss / dataset_sizes[phase]
            try:
                epoch_auc_ave = roc_auc_score(np.array(label_list), np.array(output_list))
                epoch_auc = roc_auc_score(np.array(label_list), np.array(output_list), average=None)
            except:
                epoch_auc_ave = 0
                epoch_auc = [0 for _ in range(len(class_names))]

            if phase == 'val':
                writer.add_scalar('loss/validation', epoch_loss, epoch)
                writer.add_scalar('auc/validation', epoch_auc_ave, epoch)
            if phase == 'train':
                writer.add_scalar('loss/train', epoch_loss, epoch)
                writer.add_scalar('auc/train', epoch_auc_ave, epoch)

            log_str = ''
            log_str += 'Loss: {:.4f} AUC: {:.4f}  \n\n'.format(
                epoch_loss, epoch_auc_ave)

            for i, c in enumerate(class_names):
                log_str += '{}: {:.4f}  \n'.format(c, epoch_auc[i])

            log_str += '\n'
            if phase == 'val':
                print("\n\nValidation Phase ")
            else:
                print("\n\nTraining Phase ")

            print(log_str)
            writer.add_text('log', log_str, iter_num)
            print("Best validation average AUC :", best_auc_ave)
            print("Average AUC of current epoch :", epoch_auc_ave)

            # save model for validation
            if phase == 'val' and epoch_auc_ave > best_auc_ave:
                best_auc = epoch_auc
                print("Rewriting model with AUROC :", round(best_auc_ave, 4), " by model with AUROC : ",
                      round(epoch_auc_ave, 4))
                best_auc_ave = epoch_auc_ave
                print('Model saved to %s' % bestmodel_file)
                print("Saving the best checkpoint")
                SaveModel(epoch, model, optimizer, best_auc_ave, bestmodel_file)

            if phase == 'train' and epoch % 1 == 0:
                SaveModel(epoch, model, optimizer, best_auc_ave, checkpoint_file)

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


def training_abnormal(model, args):
    print("_____________________________")
    print("__________Binary_____________")
    print("_____________________________")
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, model_name(args)))
    optimizer = get_optimizer(model.parameters(), args)
    dataloaders, dataset_sizes, class_names = make_dataLoader_binary(args)
    criterion = torch.nn.BCELoss()

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    best_auc_ave = 0.0  # to check if best validation accuracy
    epoch_inti = 1  # epoch starts from here

    best_model_wts = model.state_dict()
    iter_num = 0

    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(args.model_save_dir, str(model_name(args) + '_' + "checkpoint.pth"))
    bestmodel_file = os.path.join(args.model_save_dir, str(model_name(args) + '_' + "best_model.pth"))

    ''' Check for existing training results. If it existst, and the configuration
    is set to resume `config.resume_TIRG==True`, resume from previous training.
    If not, delete existing checkpoint.'''
    if os.path.exists(checkpoint_file):
        if args.resume:
            model, optimizer, epoch_inti, best_auc_ave = LoadModel(checkpoint_file, model, optimizer, epoch_inti,
                                                                   best_auc_ave)
            print("Checkpoint found! Resuming")
        else:
            pass

    since = time.time()

    for epoch in range(epoch_inti, args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        # scheduler.step()
        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            # Only Validation in every 5 cycles
            if (phase == 'val') and (epoch % 1 != 0):
                continue
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            output_list = []
            label_list = []
            loss_list = []
            # Iterate over data.
            for idx, data in enumerate(tqdm(dataloaders[phase])):

                # if idx == 1000:
                #     break
                images, labels, names = data
                images = images.to(device)
                labels = labels.to(device)
                # labels = (labels.sum(axis=1) >= 1) * 1
                labels = labels.unsqueeze(axis=1)

                # calculate weight for loss
                P, N = 0, 0
                for label in labels:
                    for v in label:
                        if int(v) == 1:
                            P += 1
                        else:
                            N += 1
                if P != 0 and N != 0:
                    BP = (P + N) / P
                    BN = (P + N) / N
                    weights = torch.tensor([BP, BN], dtype=torch.float).to(device)
                else:
                    weights = None

                if phase == 'train':
                    torch.set_grad_enabled(True)
                else:
                    torch.set_grad_enabled(False)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(images)

                # classification loss
                loss = weighted_BCELoss(outputs, labels, weights=weights)
                # loss = criterion(outputs, labels.type(torch.cuda.FloatTensor))
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    iter_num += 1

                running_loss += loss.item()
                loss_list.append(loss.item())
                outputs = outputs.detach().to('cpu').numpy()
                labels = labels.detach().to('cpu').numpy()

                for i in range(outputs.shape[0]):
                    output_list.append(outputs[i].tolist())
                    label_list.append(labels[i].tolist())

                if idx % 10 == 0:
                    if phase == 'train':
                        writer.add_scalar('Loss/Train', loss.item() / outputs.shape[0], iter_num)
                if idx % 100 == 0 and idx != 0:
                    if phase == 'train':
                        try:
                            auc = roc_auc_score(np.array(label_list[-100 * args.batch_size:]),
                                                np.array(output_list[-100 * args.batch_size:]))
                            print('\nAUC/Train', auc)
                            print('Batch Loss', sum(loss_list) / len(loss_list))
                            print('Batch Accuracy', ((((np.array(output_list) > 0.5) * 1)
                                                      == np.array(label_list)) * 1).mean())
                            loss_list = []
                            writer.add_scalar('AUC/Train', auc, iter_num)
                        except:
                            pass
            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_auc_ave = roc_auc_score(np.array(label_list), np.array(output_list))

            if phase == 'val':
                writer.add_scalar('Loss/Validation', epoch_loss, iter_num)
                writer.add_scalar('AUC/Validation', epoch_auc_ave, iter_num)

            log_str = ''
            log_str += 'Loss: {:.4f} AUC: {:.4f}  \n\n'.format(epoch_loss, epoch_auc_ave)
            log_str += '\n'
            if phase == 'val':
                print("\n\nValidation Phase ")
            else:
                print("\n\nTraining Phase ")

            print(log_str)
            writer.add_text('log', log_str, iter_num)
            print("Best average AUC :", best_auc_ave)
            print("Average AUC of current epoch", epoch_auc_ave)

            # save model for validation
            if phase == 'val' and epoch_auc_ave > best_auc_ave:
                print("Rewriting model with accuracy", round(best_auc_ave, 4), " by ", round(epoch_auc_ave, 4))
                best_auc_ave = epoch_auc_ave
                print('Model saved to %s' % bestmodel_file)
                print("Saving the best checkpoint")
                SaveModel(epoch, model, optimizer, best_auc_ave, bestmodel_file)
                writer.add_text('log', 'Model saved to %s\n\n' % (args.model_save_dir), iter_num)

            if phase == 'train' and epoch % 1 == 0:
                SaveModel(epoch, model, optimizer, best_auc_ave, checkpoint_file)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AUC: {:4f}'.format(best_auc_ave))
    print()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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


def training_PCAM(model, args):
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, model_name(args)))
    writer.add_text('log', str(args), 0)
    if args.backbone == 'resnet50_wildcat':
        params = model.get_config_optim(args.lr, 0.5)
    else:
        params = model.parameters()
    optimizer = get_optimizer(params, args)
    dataloaders, dataset_sizes, class_names = make_dataLoader(args)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    best_auc_ave = 0.0  # to check if best validation accuracy
    epoch_inti = 1  # epoch starts from here

    best_model_wts = model.state_dict()
    best_auc = []
    iter_num = 0

    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(args.model_save_dir, str(model_name(args) + '_' + "checkpoint.pth"))
    bestmodel_file = os.path.join(args.model_save_dir, str(model_name(args) + '_' + "best_model.pth"))

    ''' Check for existing training results. If it existst, and the configuration
    is set to resume `config.resume_TIRG==True`, resume from previous training. 
    If not, delete existing checkpoint.'''
    if os.path.exists(checkpoint_file):
        if args.resume:
            model, optimizer, epoch_inti, best_auc_ave = LoadModel(checkpoint_file, model, optimizer, epoch_inti,
                                                                   best_auc_ave)
            print("Checkpoint found! Resuming")
        else:
            pass

    print(model)
    since = time.time()

    for epoch in range(epoch_inti, args.epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            # Only Validation in every 5 cycles
            if (phase == 'val') and (epoch % 1 != 0):
                continue
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            output_list = []
            label_list = []
            lr = lr_schedule(args.lr, 0.1, epoch,
                             [4, 8, 12])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # Iterate over data.
            for idx, data in enumerate(tqdm(dataloaders[phase])):

                images, labels, names = data
                images = images.to(device)
                labels = labels.to(device)

                if phase == 'train':
                    torch.set_grad_enabled(True)
                else:
                    torch.set_grad_enabled(False)

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs, logit_map = model(images)

                # classification loss
                loss = 0
                for t in range(args.num_classes):
                    loss_t, acc_t = get_loss(outputs, labels, t, device, args)
                    loss += loss_t

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    iter_num += 1

                running_loss += loss.item()
                outputs = outputs.detach().to('cpu').numpy()
                labels = labels.detach().to('cpu').numpy()

                for i in range(outputs.shape[0]):
                    output_list.append(outputs[i].tolist())
                    label_list.append(labels[i].tolist())

                if idx % 100 == 0 and idx != 0:
                    if phase == 'train':
                        writer.add_scalar('loss/train_batch', loss.item() / outputs.shape[0], iter_num)
                        try:
                            auc = roc_auc_score(np.array(label_list[-100 * args.batch_size:]),
                                                np.array(output_list[-100 * args.batch_size:]))
                            writer.add_scalar('auc/train_batch', auc, iter_num)
                        except:
                            pass

            epoch_loss = running_loss / dataset_sizes[phase]
            try:
                epoch_auc_ave = roc_auc_score(np.array(label_list), np.array(output_list))
                epoch_auc = roc_auc_score(np.array(label_list), np.array(output_list), average=None)
            except:
                epoch_auc_ave = 0
                epoch_auc = [0 for _ in range(len(class_names))]

            if phase == 'val':
                writer.add_scalar('loss/validation', epoch_loss, epoch)
                writer.add_scalar('auc/validation', epoch_auc_ave, epoch)
            if phase == 'train':
                writer.add_scalar('loss/train', epoch_loss, epoch)
                writer.add_scalar('auc/train', epoch_auc_ave, epoch)

            log_str = ''
            log_str += 'Loss: {:.4f} AUC: {:.4f}  \n\n'.format(
                epoch_loss, epoch_auc_ave)

            for i, c in enumerate(class_names):
                log_str += '{}: {:.4f}  \n'.format(c, epoch_auc[i])

            log_str += '\n'
            if phase == 'val':
                print("\n\nValidation Phase ")
            else:
                print("\n\nTraining Phase ")

            print(log_str)
            writer.add_text('log', log_str, iter_num)
            print("Best validation average AUC :", best_auc_ave)
            print("Average AUC of current epoch :", epoch_auc_ave)

            # save model for validation
            if phase == 'val' and epoch_auc_ave > best_auc_ave:
                best_auc = epoch_auc
                print("Rewriting model with AUROC :", round(best_auc_ave, 4), " by model with AUROC : ",
                      round(epoch_auc_ave, 4))
                best_auc_ave = epoch_auc_ave
                print('Model saved to %s' % bestmodel_file)
                print("Saving the best checkpoint")
                SaveModel(epoch, model, optimizer, best_auc_ave, bestmodel_file)

            if phase == 'train' and epoch % 1 == 0:
                SaveModel(epoch, model, optimizer, best_auc_ave, checkpoint_file)

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
    args = parser.parse_args()

    # print("\n\n Configrations \n Backbone : {} \n Attention used :{} \n Number of classes : {}"
    #       "\n Global Pooling method :{} \n\n".format(args.backbone, args.attention_map, args.num_classes,
    #                                                  args.global_pool))
    # model = select_model(args)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = DataParallel(model)
    # model.to(device)
    # # training_abnormal(model, args)
    # training(model, args)
    #
    # args.global_pool = 'MAX'
    # print("\n\n Configrations \n Backbone : {} \n Attention used :{} \n Number of classes : {}"
    #       "\n Global Pooling method :{} \n\n".format(args.backbone, args.attention_map, args.num_classes,
    #                                                  args.global_pool))
    # model = select_model(args)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = DataParallel(model)
    # model.to(device)
    # # training_abnormal(model, args)
    # training(model, args)
    #
    # args.global_pool = 'LSE'
    # print("\n\n Configrations \n Backbone : {} \n Attention used :{} \n Number of classes : {}"
    #       "\n Global Pooling method :{} \n\n".format(args.backbone, args.attention_map, args.num_classes,
    #                                                  args.global_pool))
    # model = select_model(args)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = DataParallel(model)
    # model.to(device)
    # # training_abnormal(model, args)
    # training(model, args)
    # args.backbone = "EfficientNet"
    # args.batch_size = 32
    # print("\n\n Configrations \n Backbone : {} \n Attention used :{} \n Number of classes : {}"
    #       "\n Global Pooling method :{} \n\n".format(args.backbone, args.attention_map, args.num_classes,
    #                                                  args.global_pool))
    # model = select_model(args)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = DataParallel(model)
    # model.to(device)
    # # training_abnormal(model, args)
    # training(model, args)

    # args.pretrained = True
    # print("\n\n Configrations \n Backbone : {} \n Pretrained weights : {} \n Attention used :{}"
    #       " \n Number of classes : {} \n Global Pooling method :{} \n\n"
    #       .format(args.backbone, str(args.pretrained),  args.attention_map, args.num_classes, args.global_pool))
    # model = select_model(args)
    # model = training_abnormal(args)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = DataParallel(model)
    # model.to(device)
    # training_PCAM(model, args)
    # model = eval_function(args, model)

    print("\n\n Configrations \n Backbone : {} \n Pretrained weights : {} \n Attention used :{}"
          " \n Number of classes : {} \n Global Pooling method :{} \n\n"
          .format(args.backbone, str(args.pretrained), args.attention_map, args.num_classes, args.global_pool))
    model = select_model(args)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)
    model.to(device)
    model = training_abnormal(model, args)
    eval_function(args, model)
