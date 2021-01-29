import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.patches as patches
from X_RAY.cxr8.utils import visualize_cam
import os
from train import parser, select_model
from dataset import CXRDataset

args = parser.parse_args()
current_dict = os.getcwd()

# see activations.py for descriptions
test_txt_path = "/home/ubuntu/project/data/test_bbox_list.txt"

data_root_dir = os.path.join(current_dict, 'dataset')
gcam_outputs_path = os.path.join(current_dict, "activation_maps/gcam_output.npy")
gradcam_masks_path = os.path.join(current_dict, "activation_maps/gradcam_masks.npy")
gradcam_heatmaps_path = os.path.join(current_dict, "activation_maps/gradcam_heatmaps.npy")
gradcam_result_paths = os.path.join(current_dict, "activation_maps/gradcam_results.npy")
gradcampp_masks_path = os.path.join(current_dict, "activation_maps/gradcampp_masks.npy")
gradcampp_heatmaps_path = os.path.join(current_dict, "activation_maps/gradcampp_heatmaps.npy")
gradcampp_results_path = os.path.join(current_dict, "activation_maps/gradcampp_results.npy")
image_id_path = os.path.join(current_dict, "activation_maps/image_id.npy")
output_class_path = os.path.join(current_dict, "activation_maps/output_class.npy")


def IOU(xywh1, xywh2):  # intersection over union for two bounding boxes
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2

    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    intersection = dx * dy if (dx >= 0 and dy >= 0) else 0.

    union = w1 * h1 + w2 * h2 - intersection
    return (intersection / union)


def contains(xywh1, xywh2):  # returns True if xywh2 is completely inside xywh1
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2

    if x2 < x1:
        return False
    if y2 < y1:
        return False
    if x2 + w2 > x1 + w1:
        return False
    if y2 + h2 > y1 + h1:
        return False
    return True


def small_to_large(x1, y1, w1, h1):  # convert a 224x224 array (which is center-cropped from 256x256) to 1024x1024 array
    x2 = x1 + 16
    y2 = y1 + 16
    x2 = x2 * 4
    y2 = y2 * 4
    w2 = w1 * 4
    h2 = h1 * 4
    return x2, y2, w2, h2


def large_to_small(x1, y1, w1, h1,
                   cropped=True):  # convert 1024x1024 to 224x224 (which is center-cropped from 256x256)
    x2 = x1 / 4
    y2 = y1 / 4
    w2 = w1 / 4
    h2 = h1 / 4
    if cropped:
        if x2 < 16:
            x2 = 0
            w2 = w2 - 16
        else:
            x2 = x2 - 16
        if x2 + w2 > 224:
            w2 = 224 - x2
        if y2 < 16:
            y2 = 0
            h2 = h2 - 16
        else:
            y2 = y2 - 16
        if y2 + h2 > 224:
            h2 = 224 - y2
    return int(x2), int(y2), int(w2), int(h2)


def main():
    print("beginning...")
    args = parser.parse_args()
    # test_X = np.load("/home/ubuntu/project/data/postproc/test_bbox_X_small.npy")
    # with open("/home/ubuntu/project/data/postproc/test_bbox_y_onehot.pkl", "rb") as f:
    #     test_y = pickle.load(f)

    trans = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    datasets = CXRDataset(data_root_dir, dataset_type='box', Num_classes=args.num_classes,
                          img_size=args.img_size, transform=trans)
    dataloaders = DataLoader(datasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # class ChestXrayDataSet_plot(Dataset):
    #     def __init__(self, input_X=test_X, transform=None):
    #         self.X = np.uint8(test_X * 255)
    #         self.transform = transform
    #
    #     def __getitem__(self, index):
    #         """
    #         Args:
    #             index: the index of item
    #         Returns:
    #             image
    #         """
    #         current_X = np.tile(self.X[index], 3)
    #         image = self.transform(current_X)
    #         return image  # (3,224,224)
    #
    #     def __len__(self):
    #         return len(self.X)
    #
    # test_dataset = ChestXrayDataSet_plot(input_X=test_X, transform=transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ]))
    # with open(test_txt_path, "r") as f:
    #     test_list = [i.strip() for i in f.readlines()]
    print("test dataset loaded")
    print("number of test examples:", len(dataloaders))
    print("loading class activation maps")
    print(gcam_outputs_path)
    gcam_outputs = np.load(gcam_outputs_path)  # (2244, 224, 224)
    gradcam_masks = np.load(gradcam_masks_path)  # (2244, 224, 224)
    gradcam_heatmaps = np.load(gradcam_heatmaps_path)  # (2244, 3, 224, 224)
    gradcam_results = np.load(gradcam_result_paths)  # (2244, 3, 224, 224)

    gradcampp_masks = np.load(gradcampp_masks_path)
    gradcampp_heatmaps = np.load(gradcampp_heatmaps_path)
    gradcampp_results = np.load(gradcampp_results_path)
    image_ids = np.load(image_id_path)
    output_classes = np.load(output_class_path)

    thresholds = np.load("thresholds.npy")  # Youden indices for each class
    print("activate threshold", thresholds)


    # ---- Initialize the network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_GPU = torch.cuda.device_count()
    args = parser.parse_args()
    model = select_model(args)

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model

    current_location = os.getcwd()
    pathModel = os.path.join(current_location, "savedModels", "densenet121_LSE_IMG_SIZE_224num_class_8_best_model.pth")
    modelCheckpoint = torch.load(pathModel)
    if num_GPU > 1:
        model.module.load_state_dict(modelCheckpoint['model_state_dict'])
    else:
        model.load_state_dict(modelCheckpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("model loaded")
    bbox_df = pd.read_csv(os.path.join(data_root_dir, 'BBox_List_2017.csv') )
    print("bounding boxes loaded")

    see_heatmaps = False  # change to True to see class activation maps for an image
    if see_heatmaps:  # heatmaps/results for activation maps
        for i , (img,label, bbox, name) in enumerate(dataloaders) :  # test image
            if i == 10 :
                break
            img_temp = img.cuda()
            name = name[0].split('.')[0]
            img = img.squeeze().permute([1, 2, 0])
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
            plt.savefig(os.path.join(current_dict,("heatmap_output/img_" + name + ".png")))

            probs = model(img_temp).cpu().data.numpy()
            activate_classes = np.where((probs > thresholds)[0] == True)[0]
            heatmaps = np.where(image_ids == i)[0]
            print(heatmaps)
            # remember, we have three grad-cam implementations:
            # gcam, a T.H. Tang implementation
            # gradcam, a Won Kwang Lee implementation (didn't work well because of too many NaN's)
            # gradcam++, a Won Kwang Lee implementation
            for j in heatmaps:
                c = output_classes[j]
                plt.imshow(gcam_outputs[j])
                plt.savefig(os.path.join(current_dict,("heatmap_output/gcam_" + name + "_class_" + str(c) + ".png")))
                _, gcam_result = visualize_cam(torch.Tensor(gcam_outputs[j]).unsqueeze(0).unsqueeze(0),
                                               img.permute([2, 0, 1]).unsqueeze(0))
                gcam_result = gcam_result.permute([1, 2, 0])
                plt.imshow(gcam_result)
                plt.savefig(
                    os.path.join(current_dict,("heatmap_output/gcam_result_" + name + "_class_" + str(c) + ".png")))
                plt.imshow(gradcam_masks[j])
                plt.savefig(
                    os.path.join(current_dict,("heatmap_output/gradcam_mask_" + name + "_class_" + str(c) + ".png")))
                plt.imshow(gradcampp_masks[j])
                plt.savefig(
                    os.path.join(current_dict,("heatmap_output/gradcampp_mask_" + name + "_class_" + str(c) + ".png")))
                plt.imshow(gradcam_heatmaps[j].transpose([1, 2, 0]))
                plt.savefig(
                    os.path.join(current_dict,("heatmap_output/gradcam_heatmap_" + name + "_class_" + str(c) + ".png")))
                plt.imshow(gradcampp_heatmaps[j].transpose([1, 2, 0]))
                plt.savefig(
                    os.path.join(current_dict,("heatmap_output/gradcampp_heatmap_" + name + "_class_" + str(c) + ".png")))
                gradcam_result = gradcam_results[j].transpose([1, 2, 0])
                gradcam_result = (gradcam_result - gradcam_result.min()) / (gradcam_result.max() - gradcam_result.min())
                plt.imshow(gradcam_result)
                plt.savefig(
                     os.path.join(current_dict,("heatmap_output/gradcam_results_" + name + "_class_" + str(c) + ".png")))
                gradcampp_result = gradcampp_results[j].transpose([1, 2, 0])
                gradcampp_result = (gradcampp_result - gradcampp_result.min()) / (
                        gradcampp_result.max() - gradcampp_result.min())
                plt.imshow(gradcampp_result)
                plt.savefig(
                     os.path.join(current_dict,("heatmap_output/gradcampp_results_" + name + "_class_" + str(c) + ".png")))

    class_index = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax']
    outputs = []

    bbox_predictions = []
    ious = []
    contain_list = []
    plot = True
    for i, (image, label, bbox, name) in enumerate(dataloaders):  # per image in test

        name = name[0].split('.')[0]
        gt_x = bbox[0][0]  # ground truths
        gt_y = bbox[0][1]
        gt_w = bbox[0][2]
        gt_h = bbox[0][3]

        img = image.squeeze().permute([1, 2, 0])
        img = (img - img.min()) / (img.max() - img.min())
        sm_x, sm_y, sm_w, sm_h = large_to_small(gt_x, gt_y, gt_w, gt_h, cropped=True)  # 224 x224
        if plot:
            plt.imshow(img)  # ground truth
            ax = plt.gca()
            [p.remove() for p in reversed(ax.patches)]
            rect = patches.Rectangle((sm_x, sm_y), sm_w, sm_h,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.savefig(os.path.join(current_dict, "bounding_box_images/gt_cropped_boundingbox_" + name + ".png"))
            plt.close()

        # gradcampp
        heatmaps = np.where(image_ids == i)[0]
        labels = []
        indices = []
        max_activations = []
        for j in heatmaps:  # for all activation maps (different classes) for an image
            activations = gradcam_masks[j]
            if np.isnan(activations).any():  # get rid of any NaNs
                print("NaNs!")
                continue
            thr = activations.mean() * 1.95
            if thr > 0.6:
                thr = 0.6
            mask = activations > thr  # hyperparameter?
            label_im, nb_labels = ndimage.label(mask)
            sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
            index = np.argmax(sizes)
            labels.append(label_im)
            indices.append(index)
            max_activations.append(sizes[index])

        if len(max_activations) == 0:  # if none of the heatmaps work
            ious.append(0)
            contain_list.append(False)
            print("NO ACTIVATIONS FOR " + str(i))
            continue
        max_activation_index = np.argmax(max_activations)
        index = indices[max_activation_index]
        label_im = labels[max_activation_index]
        slice_y, slice_x = ndimage.find_objects(label_im == index)[0]

        # predictions
        pr_x = slice_x.start
        pr_y = slice_y.start
        pr_w = slice_x.stop - slice_x.start
        pr_h = slice_y.stop - slice_y.start
        if plot:
            ax = plt.gca()  # prediction
            [p.remove() for p in reversed(ax.patches)]
            ax.imshow(img)
            rect = patches.Rectangle((pr_x, pr_y), pr_w, pr_h,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.savefig(os.path.join(current_dict,"bounding_box_images/gradcampp_thr_boundingbox_" + name + ".png"))
            plt.close()

            ax = plt.gca()
            [p.remove() for p in reversed(ax.patches)]
            ax.imshow(img)  # prediction AND ground truth
            rect1 = patches.Rectangle((pr_x, pr_y), pr_w, pr_h,  # prediction
                                      linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect1)
            rect2 = patches.Rectangle((sm_x, sm_y), sm_w, sm_h,  # ground truth
                                      linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect2)
            plt.savefig(os.path.join(current_dict,"bounding_box_images/gradcampp_both_boundingbox_" + name + ".png"))
            plt.close()

        bbox_predictions.append([pr_x, pr_y, pr_w, pr_h])
        iou = IOU((pr_x, pr_y, pr_w, pr_h), (sm_x, sm_y, sm_w, sm_h))
        contain = contains((pr_x, pr_y, pr_w, pr_h), (sm_x, sm_y, sm_w, sm_h))
        ious.append(iou)
        contain_list.append(contain)
        print(str(i) + "       " + str(iou) + "         " + str(contain))

    ious = np.array([ious]).squeeze()
    contain_list = np.array([contain_list]).squeeze()
    print("mean IOU: " + str(ious.mean()))
    print("incorrect: " + str(len(np.where(ious == 0.0)[0])))
    print(len(np.where(ious == 0.0)[0]) / len(contain_list))
    print("contains: " + str(contain_list.sum()))
    print(contain_list.sum() / len(contain_list))
    for c in class_index:
        class_indices = bbox_df.loc[bbox_df['Finding Label'] == c].index.tolist()

        class_iou = ious[class_indices].mean()
        contain = contain_list[class_indices].sum()
        print(c + " total        " + str(len(class_indices)))
        print(c + " iou          " + str(class_iou))
        print(c + " contains     " + str(contain / len(class_indices)))


if __name__ == '__main__':
     main()
