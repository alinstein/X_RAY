import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from sklearn.metrics import roc_auc_score
import numpy as np
import time
from tqdm import tqdm
import os

from utlis.utils import model_name, select_model
from utlis.utils import get_optimizer, make_dataLoader, LoadModel, lr_schedule, make_dataLoader_chexpert
from utlis.utils import weighted_BCELoss, SaveModel, make_dataLoader_binary, get_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training(model, args):
    """
    This function trains the multi-label model.
    model: PyTorch model with model
    args: configuration file (argparse)
    return: train the returned the model
    """

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, model_name(args)))
    writer.add_text('log', str(args), 0)
    if args.backbone == 'resnet50_wildcat':
        params = model.get_config_optim(args.lr, 0.5)
    else:
        params = model.parameters()
    optimizer = get_optimizer(params, args)
    if args.dataset == "NIH":
        dataloaders, dataset_sizes, class_names =  make_dataLoader(args)
    elif args.dataset == 'ChesXpert':
        dataloaders, dataset_sizes, class_names = make_dataLoader_chexpert(args)
    else:
        assert "Wrong dataset"

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
        iter_num = 0
        for phase in ['train','val']:

            # Only Validation in every 5 cycles
            if (phase == 'val') :
                model.train(False)
            elif phase == 'train':
                model.train(True)  # Set model to training mode


            running_loss = 0.0
            output_list = []
            label_list = []
            loss_list = []

            # Iterate over data.
            for idx, data in enumerate(tqdm(dataloaders[phase])):

                # if iter_num > 100 and phase == 'train':
                #     break

                images, labels, names = data
                images = images.to(device)
                labels = labels.to(device)

                mask = 1 * (labels >= 0)
                mask = mask.to(device)

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
                # Predicting output
                outputs = model(images)

                # classification loss
                if args.weighted_loss:
                    loss = weighted_BCELoss(outputs, labels, weights=weights)
                else:
                    loss_func = torch.nn.BCELoss(reduction='none')
                    loss = loss_func(outputs[:, 0].unsqueeze(dim=-1),
                                     torch.tensor(labels[:, 0].unsqueeze(dim=-1), dtype=torch.float))
                    loss = torch.where((labels >= 0)[:, 0].unsqueeze(axis=-1), loss, torch.zeros_like(loss)).mean()
                    for index in range(1, args.num_classes):
                        loss_temp = loss_func(outputs[:, index].unsqueeze(dim=-1),
                                              torch.tensor(labels[:, index].unsqueeze(dim=-1), dtype=torch.float))
                        loss_temp = torch.where((labels >= 0)[:, index].unsqueeze(axis=-1), loss_temp,
                                                torch.zeros_like(loss_temp)).mean()
                        loss += loss_temp

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
                    output_list.append(np.where(labels[i] >= 0, outputs[i], 0).tolist())
                    label_list.append(np.where(labels[i] >= 0, labels[i], 0).tolist())

                # Saving logs
                if idx % 100 == 0 and idx != 0:
                    if phase == 'train':
                        writer.add_scalar('loss/train_batch', loss.item() / outputs.shape[0], iter_num)
                        try:
                            auc = roc_auc_score(np.array(label_list[-100 * args.batch_size:]),
                                                np.array(output_list[-100 * args.batch_size:]))
                            writer.add_scalar('auc/train_batch', auc, iter_num)
                            print('\nAUC/Train', auc)
                            print('Batch Loss', sum(loss_list) / len(loss_list))
                        except:
                            pass

            epoch_loss = running_loss / dataset_sizes[phase]
            # Computing AUC
            try:
                epoch_auc_ave = roc_auc_score(np.array(label_list), np.array(output_list))
                epoch_auc = roc_auc_score(np.array(label_list), np.array(output_list), average=None)
            except ValueError:
                epoch_auc_ave = roc_auc_score(np.array(label_list)[:, :12], np.array(output_list)[:, :12]) * (12 / 13) + \
                                roc_auc_score(np.array(label_list)[:, 13], np.array(output_list)[:, 13]) * (1 / 13)
                epoch_auc = roc_auc_score(np.array(label_list)[:, :12], np.array(output_list)[:, :12], average=None)
                epoch_auc = np.append(epoch_auc, 0)
                epoch_auc = np.append(epoch_auc, roc_auc_score(np.array(label_list)[:, 13], np.array(output_list)[:, 13], average=None))

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

            # save model with best validation AUC
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
    try:
        for i, c in enumerate(class_names):
            print('{}: {:.4f} '.format(c, best_auc[i]))
    except:
        for i, c in enumerate(class_names):
            print('{}: {:.4f} '.format(c, epoch_auc[i]))


    # load best model weights to return
    model.load_state_dict(best_model_wts)

    return model


def training_abnormal(model, args):
    """
    Train the binary classification model.
    model: PyTorch model with model
    args: configuration file (argparse)
    return: train the returned the model
    """
    print("____________________________________________")
    print("__________Binary Classification_____________")
    print("____________________________________________")
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

                images, labels, names = data
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(axis=1)
                # if labels.sum() == 0:
                #     continue

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


def training_PCAM(model, args):
    """
    Train the PCAM model.
    model: PyTorch model with model
    args: configuration file (argparse)
    return: train the returned the model
    """
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
