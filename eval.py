import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, hamming_loss
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.dataset import CXRDataset, CXRDatasetBinary
from utlis.utils import model_name, select_model
from config import parser



def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def compute_stats(gt, pred, cfg):
    if cfg.num_classes == 1:
        CLASS_NAMES = ['Disease']
    else:
        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                       'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
    N_CLASSES = len(CLASS_NAMES)
    AUROCs = []
    roc_curves = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()

    for i in range(N_CLASSES):
        if cfg.num_classes == 1:
            AUROCs.append(roc_auc_score(gt_np, pred_np))
            roc_curves.append(roc_curve(gt_np, pred_np))
        else:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
            roc_curves.append(roc_curve(gt_np[:, i], pred_np[:, i]))

    TPR_list = []
    TNR_list = []
    PPV_list = []

    TPR_dict = {}
    TNR_dict = {}
    PPV_dict = {}
    Hamming_loss = []
    for i in range(N_CLASSES):
        if cfg.num_classes == 1:
            opt_threh = Find_Optimal_Cutoff(gt_np, pred_np)
            confusion_matrix_ = confusion_matrix(gt_np.astype(int), pred_np >= opt_threh)
            Hamming_loss.append(hamming_loss(gt_np.astype(int), pred_np >= opt_threh))
        else:
            opt_threh = Find_Optimal_Cutoff(gt_np[:, i], pred_np[:, i])
            confusion_matrix_ = confusion_matrix(gt_np[:, i].astype(int), pred_np[:, i] >= opt_threh)
            Hamming_loss.append(hamming_loss(gt_np[:, i].astype(int), pred_np[:, i] >= opt_threh))
        FP = (confusion_matrix_.sum(axis=0) - np.diag(confusion_matrix_))
        FN = (confusion_matrix_.sum(axis=1) - np.diag(confusion_matrix_))
        TP = np.diag(confusion_matrix_)
        TN = (confusion_matrix_.sum() - (FP + FN + TP))

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        TPR_list.append(TPR)
        TPR_dict[CLASS_NAMES[i]] = TPR
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        TNR_list.append(TNR)
        TNR_dict[CLASS_NAMES[i]] = TNR
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        PPV_list.append(PPV)
        PPV_dict[CLASS_NAMES[i]] = PPV

    mean_TPR = sum(np.array(TPR_list)[:, 1]) / N_CLASSES
    mean_TNR = sum(np.array(TNR_list)[:, 1]) / N_CLASSES
    mean_PPV = sum(np.array(PPV_list)[:, 1]) / N_CLASSES
    mean_Hamming_loss = sum(Hamming_loss) / N_CLASSES
    return AUROCs, roc_curves, mean_TPR, mean_TNR, mean_PPV, PPV_dict, TNR_dict, TPR_dict, mean_Hamming_loss


def eval_function(args, model):
    curve_path = "ROC Curves/"
    if args.num_classes == 1:
        CLASS_NAMES = ['Disease']
    else:
        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                       'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
    trans = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    current_location = os.getcwd()
    data_root_dir = os.path.join(current_location, 'dataset')
    if args.multi_label:
        datasets = CXRDataset(data_root_dir, dataset_type='test', Num_classes=args.num_classes,
                              img_size=args.img_size, transform=trans)
    else:
        datasets = CXRDatasetBinary(data_root_dir, dataset_type='test1',
                                    img_size=args.img_size, transform=trans)

    dataloader = DataLoader(datasets, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("beginning...")
    N_CLASSES = args.num_classes
    print("test dataset loaded")
    cudnn.benchmark = True

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()  # of shape (# of batches * batch_size, 8)
    pred = torch.FloatTensor().cuda()
    print("testing...")
    test_length = len(datasets)
    print("total test examples: " + str(test_length))
    print("total batches: " + str(int(test_length / args.batch_size)))

    for i, (inputs, target, weight) in tqdm(enumerate(dataloader), total=int(test_length / args.batch_size)):
        target = target.cuda()
        inputs = inputs.to(device)
        gt = torch.cat((gt, target), 0)
        with torch.no_grad():
            if args.global_pool == "PCAM":
                output, _ = model(inputs)
                output = torch.sigmoid(output)
            else:
                output = model(inputs)
        pred = torch.cat((pred, output.data), 0)

    AUROCs, roc_curves, mean_TPR, mean_TNR, mean_PPV, PPV_dict, TNR_dict, TPR_dict, mean_Hamming_loss \
        = compute_stats(gt, pred, args)
    AUROC_avg: None = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
    print("Mean hamming loss is {}".format(mean_Hamming_loss))
    print("Micro-averaging Precison is {} ".format(mean_PPV))
    print("Micro-averaging Recall or Sensitivity is {} ".format(mean_TPR))
    print("Micro-averaging Specificity is {} ".format(mean_TNR))

    for i in range(N_CLASSES):
        print('The Precison of {} is {}'.format(CLASS_NAMES[i], PPV_dict[CLASS_NAMES[i]]))
    for i in range(N_CLASSES):
        print('The Recall or Sensitivity of {} is {}'.format(CLASS_NAMES[i], TPR_dict[CLASS_NAMES[i]]))
    for i in range(N_CLASSES):
        print('The Specificity of {} is {}'.format(CLASS_NAMES[i], TNR_dict[CLASS_NAMES[i]]))

    for i in range(N_CLASSES):
        fpr, tpr, thresholds = roc_curves[i]
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(fpr, tpr, label="model")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC CURVE: " + CLASS_NAMES[i])
        plt.savefig(curve_path + model_name(args) + CLASS_NAMES[i] + ".png")
        plt.clf()

    for i in range(N_CLASSES):
        fpr, tpr, thresholds = roc_curves[i]
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(fpr, tpr, label=CLASS_NAMES[i])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC CURVE")
    plt.savefig(curve_path + model_name(args) + ".png")
    plt.clf()


if __name__ == '__main__':
    args = parser.parse_args()
    # ---- Initialize the network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_GPU = torch.cuda.device_count()
    args = parser.parse_args()
    model = select_model(args)
    # model = PCAM_Model(args)
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model
    current_location = os.getcwd()
    pathModel = os.path.join(current_location, "savedModels", "compute",
                             "densenet121_True_LSE_IMG_SIZE_512_num_class_8_best_model.pth")
    modelCheckpoint = torch.load(pathModel)
    if num_GPU > 1:
        model.module.load_state_dict(modelCheckpoint['model_state_dict'])
    else:
        model.load_state_dict(modelCheckpoint['model_state_dict'])
    eval_function(args, model)
