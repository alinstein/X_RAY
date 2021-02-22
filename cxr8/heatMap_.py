import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot  as plt
from torch.autograd import Variable
import numpy as np
from train import parser, select_model
from dataset import CXRDataset
import matplotlib.patches as patches
from X_RAY.cxr8.gradcam import Grad_CAM, Grad_CAMpp
from model import PCAM_Model
from pandas import DataFrame
from X_RAY.cxr8.utils import visualize_cam


class HeatmapGenerator:
    """
    Calculate heatmaps with CAM, GradCAM, GradCAM++.
    Also compute the bounding boxes around the heatmap and prints the IOU.
    A simple example:
        # location of the saved CNN model
        pathModel = os.path.join(current_location, "savedModels", "compute",
                             "ResNet18_True_LSE_IMG_SIZE_512_num_class_8_best_model.pth")
        h = HeatmapGenerator(pathModel, save_plots=False, args=args)
        # generate() function generates the heatmap.
        h.generate(plot=False, factor=1.7666, threshold_high=0.7666)

    Args:
        pathModel: string containing location of saved model,
        save_plots: True if need to save generated heatmaps and heatmaps with bounding boxes
        args: input arguments about the image size, global pooling layer, DCNN.
    """
    def __init__(self, pathModel, save_plots, args):
        # names of diseases
        self.class_names = ['Atelectasis', 'Cardiomegaly',
                            'Effusion', 'Infiltrate',
                            'Mass', 'Nodule',
                            'Pneumonia', 'Pneumothorax']

        self.save_plots = save_plots
        self.save_location = os.path.join(os.getcwd(), "heatmap_output")
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_GPU = torch.cuda.device_count()
        args = parser.parse_args()
        # ---- Initialize the network
        if args.global_pool == 'PCAM':
            model = PCAM_Model(args)
        else:
            model = select_model(args)

        # load the model to multiple GPUs if needed
        model = model.to(self.device)
        torch.cuda.memory_allocated()
        torch.cuda.max_memory_allocated()
        modelCheckpoint = torch.load(pathModel)
        if num_GPU > 1:
            model.module.load_state_dict(modelCheckpoint['model_state_dict'])
        else:
            model.load_state_dict(modelCheckpoint['model_state_dict'])

        self.model = model
        self.model.eval()

        if args.backbone == "densenet121":
            model_dict = dict(type='densenet',
                              layer_name='img_model_features_norm5',
                              arch=self.model,
                              input_size=(args.img_size, args.img_size)
                              )
        elif args.backbone == "ResNet18":
            model_dict = dict(type='resnet',
                              layer_name='img_model_layer4_bottleneck1_bn2',
                              arch=self.model,
                              input_size=(args.img_size, args.img_size)
                              )

        # Function that generate the heatmap with GradCAM
        self.GradCAM = Grad_CAM(model_dict)
        # Function that generate the heatmap with GradCAM++
        self.GradCAMCPP = Grad_CAMpp(model_dict)

        # The weight of CNN are extraced for Class activation map method
        if args.global_pool == "PCAM":
            pass
        else:
            if args.backbone == "densenet121":
                self.weights = list(self.model.FF.parameters())[-2].squeeze()
            elif args.backbone == "ResNet18":
                self.weights = list(model.img_model.fc[1].parameters())[-2].squeeze()

        # Function that preprocess the input images, same preprocessing used for training CNN.
        trans = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        current_dict = os.getcwd()
        data_root_dir = os.path.join(current_dict, 'dataset')
        self.datasets = CXRDataset(data_root_dir, dataset_type='box', Num_classes=args.num_classes,
                                   img_size=args.img_size, transform=trans)
        self.dataloaders = DataLoader(self.datasets, batch_size=1, shuffle=True, num_workers=args.num_workers)
        self.iou_CAM = []
        self.iou_GradCAM = []
        self.iou_GradCAMCPP = []

    # --------------------------------------------------------------------------------
    @staticmethod
    def get_iou(pred_box, gt_box):
        """
        The function to compute IOU and IOBB score.
        Args:
            pred_box : the coordinate for predict bounding box
            gt_box :   the coordinate for ground truth bounding box

                        the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
                        the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
        return :
            iou and iobb score
        """

        x_1 = pred_box[0]
        y_1 = pred_box[1]
        x_2 = pred_box[0] + pred_box[2]
        y_2 = pred_box[1] + pred_box[3]
        pred_box[0] = x_1
        pred_box[1] = y_1
        pred_box[2] = x_2
        pred_box[3] = y_2

        area_pred_box = abs(x_1 - x_2) * abs(y_1 - y_2)

        x_1 = gt_box[0]
        y_1 = gt_box[1]
        x_2 = gt_box[0] + gt_box[2]
        y_2 = gt_box[1] + gt_box[3]
        gt_box[0] = x_1
        gt_box[1] = y_1
        gt_box[2] = x_2
        gt_box[3] = y_2
        # 1.get the coordinate of inters
        ixmin = max(pred_box[0], gt_box[0])
        ixmax = min(pred_box[2], gt_box[2])
        iymin = max(pred_box[1], gt_box[1])
        iymax = min(pred_box[3], gt_box[3])

        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)

        # 2. calculate the area of inters
        inters = iw * ih

        # 3. calculate the area of union
        uni = ((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]) +
               (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) -
               inters)

        # 4. calculate the overlaps between pred_box and gt_box
        iou = inters / uni
        # 5. calculate the intersection over the B-boxe
        iobb = inters / area_pred_box
        return iou, iobb

    @staticmethod
    def large_to_small(x1, y1, w1, h1, cropped=True, newsize=512):
        """
        Converts the coordinates of bounding boxes for original image 1024x1024
        to new BBOX coordinate corresponding new image size.
        Args:
            original BBOX coordinate and new image size.
        return: new coordinates
        """
        # convert 1024x1024 to 224x224 (which is center-cropped from 256x256)
        scale = 1024 // newsize
        x2 = x1 / scale
        y2 = y1 / scale
        w2 = w1 / scale
        h2 = h1 / scale
        if cropped and (newsize == 224):
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

    @staticmethod
    def excel(iou_list, iou_dict, iou_list_cum, iou_dic_cum, ioBB_list, ioBB_dict, test_name):
        """
        Save the IOU and IOBB details in an Excel file.
        """

        current_location = os.path.join(os.getcwd(), "outputs_logs")
        file_names = os.listdir(current_location)
        if file_names:
            new_name = max([int(os.path.splitext(file_names[i])[0][5:]) for i in range(len(file_names))  if (os.path.splitext(file_names[i])[0][0])=='t' ])+1
            new_name = "test_" + str(new_name) + ".xlsx"
        else:
            new_name = "test_1.xlsx"

        print("______________________________________")
        print("           " + str(test_name))
        print("Test name: ", new_name)
        print("______________________________________")

        metric_name = ["Mean of iou"]
        metric_value = [round(np.mean(iou_list), 3)]
        print("Mean IOU :", round(np.mean(iou_list), 3))

        # for name in iou_dict.keys():
        #     metric_name.append(str("IOU " + name))
        #     metric_value.append(round(np.mean(iou_dict[name])))

        print("\n")
        metric_name.append("Cumulative IOU")
        metric_value.append(round(np.mean(iou_list_cum), 3))
        print("Cumulative IOU :", round(np.mean(iou_list_cum), 3))

        print("\n")
        print("Mean of IoBB", round(np.mean(ioBB_list), 3))
        metric_name.append("Mean of IoBB")
        metric_value.append(round(np.mean(ioBB_list), 3))
        print("\n")

        for metric, name_m in zip([iou_dict, ioBB_dict], ["IOU", "IOBB"]):
            print("______", name_m, "______")
            for threshold in ["mean", 0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
                if threshold == 0:
                    print("No intersection")
                elif threshold == "mean":
                    print("mean IOU")
                else:
                    print(name_m, " greater than ", threshold)

                for name in metric.keys():
                    if threshold == "mean":
                        print(name, np.mean(np.array(metric[name])))
                        metric_name.append("Mean " + name_m + " for " + name)
                        metric_value.append(round(np.mean(np.array(metric[name])), 3))
                    elif threshold == 0:
                        print("No intersection or 0 IOU for ", name, np.mean(np.array(metric[name]) <= 0))
                        metric_name.append("No intersection or 0 for " + name_m)
                        metric_value.append(round(np.mean(np.array(metric[name]) <= 0), 3))
                    else:
                        # print(name_m, " greater than ", threshold)
                        print(name, np.mean(np.array(metric[name]) >= threshold))
                        metric_name.append(name_m + " greater than " + str(threshold) + " for " + name)
                        metric_value.append(round(np.mean(np.array(metric[name]) >= threshold), 3))
                print("\n\n")

        df = DataFrame({'metric_name': metric_name, 'metric_value': metric_value})
        df.to_excel(os.path.join(current_location,new_name), sheet_name='sheet1', index=False)


    def boxIouPlot(self, npHeatmap, imgOriginal, bbox, threshold_high, factor, plot, label_name, method, name):
        """
        Function return IOU, IOBB and computes the heatmap and bounding box.
        :param npHeatmap: input heatmap with shape of (1, H, W)
        :param imgOriginal: input image with shape of (3, H, W)
        :param bbox: Ground truth bbox coordinates [x1, y1, w1, h1]
        :param threshold_high :(float) Maximum threshold value
        :param factor: multiplication factor for threshold calculation
        :param plot: True if needed to plot the heatmaps and bounding boxes
        :param label_name: Name of the disease
        :param method: any from this list [GradCAM, GradCAM++, CAM]
        :return: Max IOU, Cumulative IOU and IOBB, Cumulative IOBB
        """

        cam = npHeatmap - np.min(npHeatmap)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, (self.args.img_size, self.args.img_size))
        bbox = self.large_to_small(*bbox.numpy()[0], cropped=True, newsize=self.args.img_size)

        thresh_low = cam.mean() * factor
        if thresh_low > threshold_high:
            thresh_low = threshold_high
        thresh_low = float(thresh_low)
        if plot or self.save_plots:
            plt.figure()
            plt.imshow(cam)
            ax = plt.gca()
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.title("Method : " + method + "   Disease : " + str(label_name))
            if self.save_plots:
                plt.savefig(
                    os.path.join(self.save_location, (name[0].split(".")[0] + " __" + method + "_" + label_name)))
            if plot:
                 plt.show()
        _, thresh = cv2.threshold(cam, thresh_low, 1, cv2.THRESH_BINARY)
        if plot or self.save_plots:
            plt.figure()
            plt.imshow(thresh.astype(np.uint8))
            ax = plt.gca()
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.title("Method : " + method + "   Disease : " + str(label_name))
        img, contours, hierarchy = cv2.findContours(
            (thresh.astype(np.uint8)),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # The first order of the contours
        max_iou = 0
        cum_iou = 0
        max_iobb = 0
        cum_iobb = 0
        for c_0 in contours:
            area = cv2.contourArea(c_0)
            x, y, w, h = cv2.boundingRect(c_0)
            # # # I have modified these values to make it work for attached picture
            # if not (1000 < area / ((self.args.img_size) ** 2) * ((1024) ** 2) < 404160):
            #     continue
            # if (w > (self.args.img_size * 0.75)) or (h > (self.args.img_size * 0.75)):
            #     continue
            Pred_bbox = [x, y, w, h]
            # Draw a straight rectangle with the points
            if plot:
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g',
                                         facecolor='none')
                ax.add_patch(rect)
            iou, iobb = self.get_iou(list(bbox), Pred_bbox)
            cum_iou = cum_iou + iou
            cum_iobb = cum_iobb + iobb
            if max_iou < iou:
                max_iou = iou
            if max_iobb < iobb:
                max_iobb = iobb
        if cum_iobb > 1:
            cum_iobb = 1
        if plot or self.save_plots:
            if self.save_plots:
                plt.savefig(os.path.join(self.save_location, (name[0].split(".")[0] + " _Bbox_"
                                                                  + method + "_" + label_name)))
            if plot:
                plt.show()
            plt.figure()
            heatmap, cam_result = visualize_cam(cam, imgOriginal)
            cam_result = cam_result.squeeze().permute(1, 2, 0)
            plt.imshow(cam_result.detach().cpu().numpy())
            ax = plt.gca()
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            for c_0 in contours:
                area = cv2.contourArea(c_0)
                x, y, w, h = cv2.boundingRect(c_0)
                if not (1000 < area / ((self.args.img_size) ** 2) * ((1024) ** 2) < 404160):
                    continue
                if (w > (self.args.img_size * 0.75)) or (h > (self.args.img_size * 0.75)):
                    continue
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g',
                                         facecolor='none')
                ax.add_patch(rect)
            if self.save_plots:
                plt.savefig(os.path.join(self.save_location, (name[0].split(".")[0] + " _XRay-merge_"
                                                              + method + "_" + label_name)))
            if plot:
                plt.show()
        if not contours:
            return 0, 0, 0, 0
        else:
            return max_iou, cum_iou, cum_iobb, max_iobb

    def generatePCAM(self, plot=False, factor=1.95, threshold_high=0.6):
        """

        :param plot:
        :param factor:
        :param threshold_high:
        :return:
        """

        iou_list = []
        iou_dic = {}
        for names_cl in self.class_names:
            iou_dic[names_cl] = []

        iou_list_cum = []
        iou_dic_cum = {}
        for names_cl in self.class_names:
            iou_dic_cum[names_cl] = []

        iobb_list_cum = []
        iobb_dic_cum = {}
        for names_cl in self.class_names:
            iobb_dic_cum[names_cl] = []

        with torch.no_grad():
            for idx_, (imageData, label, bbox, name, label_name) in enumerate(tqdm(self.dataloaders)):
                if torch.cuda.is_available():
                    imageData = imageData.to(self.device)
                imgOriginal = imageData.squeeze(0).detach().cpu().numpy()
                logits, logit_maps = self.model(imageData)
                label = self.class_names.index(label_name[0])
                mask = logit_maps[label]
                try:
                    iou, cum_iou, cum_iobb, max_iobb = self.boxIouPlot(mask.detach().squeeze().cpu().numpy(),
                                                                       imgOriginal,
                                                                       bbox, threshold_high, factor, plot,
                                                                       label_name[0],
                                                                       "PCAM", name)
                except:
                    pass
                iou_list.append(iou)
                iou_dic[label_name[0]].append(iou)
                iou_list_cum.append(cum_iou)
                iou_dic_cum[label_name[0]].append(cum_iou)
                iobb_list_cum.append(max_iobb)
                iobb_dic_cum[label_name[0]].append(max_iobb)

            self.excel(iou_list, iou_dic, iou_list_cum,iou_dic_cum, iobb_list_cum, iobb_dic_cum, "PCAM")
            self.iou_CAM.append(round(np.mean(iou_list_cum), 3))

    def generate(self, plot=False, factor=1.95, threshold_high=0.6):
        """
        :param plot: True if need to plot the image
        :param factor: muliplication factor for threshold
        :param threshold_high: Highest threshold value
        """

        # CAM
        iou_list = []
        iou_dic = {}
        for names_cl in self.class_names:
            iou_dic[names_cl] = []

        iou_list_cum = []
        iou_dic_cum = {}
        for names_cl in self.class_names:
            iou_dic_cum[names_cl] = []

        iobb_list_cum = []
        iobb_dic_cum = {}
        for names_cl in self.class_names:
            iobb_dic_cum[names_cl] = []

        # GradCAM
        iou_list_grad = []
        iou_dic_grad = {}
        for names_cl in self.class_names:
            iou_dic_grad[names_cl] = []

        iou_list_grad_cum = []
        iou_dic_grad_cum = {}
        for names_cl in self.class_names:
            iou_dic_grad_cum[names_cl] = []

        iobb_list_grad_cum = []
        iobb_dic_grad_cum = {}
        for names_cl in self.class_names:
            iobb_dic_grad_cum[names_cl] = []

        # GradCAM++
        iou_list_gradCPP = []
        iou_dic_gradCPP = {}
        for names_cl in self.class_names:
            iou_dic_gradCPP[names_cl] = []

        iou_list_gradCPP_cum = []
        iou_dic_gradCPP_cum = {}
        for names_cl in self.class_names:
            iou_dic_gradCPP_cum[names_cl] = []

        iobb_list_gradCPP_cum = []
        iobb_dic_gradCPP_cum = {}
        for names_cl in self.class_names:
            iobb_dic_gradCPP_cum[names_cl] = []

        with torch.no_grad():
            for idx_, (imageData, label, bbox, name, label_name) in enumerate(tqdm(self.dataloaders)):
                if torch.cuda.is_available():
                    imageData = imageData.to(self.device)
                l = self.model(imageData)

                imgOriginal = imageData.squeeze(0).detach().cpu().numpy()
                if self.args.backbone == "densenet121":
                    output = self.model.img_model.features(imageData)
                    output = F.relu(output, inplace=True)
                elif self.args.backbone == "ResNet18":
                    output = self.model.img_model.conv1(imageData)
                    output = self.model.img_model.bn1(output)
                    output = self.model.img_model.relu(output)
                    output = self.model.img_model.maxpool(output)
                    output = self.model.img_model.layer1(output)
                    output = self.model.img_model.layer2(output)
                    output = self.model.img_model.layer3(output)
                    output = self.model.img_model.layer4(output)
                    output = F.relu(output, inplace=True)

                label = self.class_names.index(label_name[0])

                # GradCAM method
                imageData_temp = Variable(imageData, requires_grad=True)
                mask, logit = self.GradCAM(imageData_temp, class_idx=label)
                try:
                    iou, cum_iou, cum_iobb, max_iobb = self.boxIouPlot(mask.detach().squeeze().cpu().numpy(),
                                                                       imgOriginal,
                                                                       bbox, threshold_high, factor, plot,
                                                                       label_name[0],
                                                                       "GradCAM", name)
                except:
                    pass
                iou_list_grad.append(iou)
                iou_dic_grad[label_name[0]].append(iou)
                iou_list_grad_cum.append(cum_iou)
                iou_dic_grad_cum[label_name[0]].append(cum_iou)
                iobb_list_grad_cum.append(max_iobb)
                iobb_dic_grad_cum[label_name[0]].append(max_iobb)

                # GradCAMCPP method
                imageData_temp = Variable(imageData, requires_grad=True)
                mask, logit = self.GradCAMCPP(imageData_temp, class_idx=label)
                try:
                    iou, cum_iou, cum_iobb, max_iobb = self.boxIouPlot(mask.detach().squeeze().cpu().numpy(),
                                                                       imgOriginal,
                                                                       bbox, threshold_high, factor, plot,
                                                                       label_name[0], "GradCAM++", name)
                except:
                    pass

                iou_list_gradCPP.append(iou)
                iou_dic_gradCPP[label_name[0]].append(iou)
                iou_list_gradCPP_cum.append(cum_iou)
                iou_dic_gradCPP_cum[label_name[0]].append(cum_iou)
                iobb_list_gradCPP_cum.append(max_iobb)
                iobb_dic_gradCPP_cum[label_name[0]].append(max_iobb)

                # CAM method
                # ---- Generate heatmap
                heatmap = None
                weights = self.weights[label]
                for i in range(0, len(weights)):
                    map = output[0, i, :, :]
                    if i == 0:
                        heatmap = weights[i] * map
                    else:
                        heatmap += weights[i] * map
                    npHeatmap = heatmap.cpu().data.numpy()
                try:
                    iou, cum_iou, cum_iobb, max_iobb = self.boxIouPlot(npHeatmap, imgOriginal, bbox,
                                                                       threshold_high, factor, plot, label_name[0],
                                                                       "CAM", name)
                except:
                    pass
                iou_list.append(iou)
                iou_dic[label_name[0]].append(iou)
                iou_list_cum.append(cum_iou)
                iou_dic_cum[label_name[0]].append(max_iobb)
                iobb_list_cum.append(cum_iobb)
                iobb_dic_cum[label_name[0]].append(max_iobb)

        self.excel(iou_list, iou_dic, iou_list_cum, iou_dic_cum, iobb_list_cum, iobb_dic_cum, "CAM")
        self.excel(iou_list_grad, iou_dic_grad, iou_list_grad_cum, iou_dic_gradCPP_cum, iobb_list_grad_cum,
                   iobb_dic_grad_cum, "GradCAM")
        self.excel(iou_list_gradCPP, iou_dic_gradCPP, iou_list_gradCPP_cum, iou_dic_gradCPP_cum,
                   iobb_list_gradCPP_cum, iobb_dic_gradCPP_cum, "GradCAM++")

        self.iou_CAM.append(round(np.mean(iou_list_cum), 3))
        self.iou_GradCAM.append(round(np.mean(iou_list_grad_cum), 3))
        self.iou_GradCAMCPP.append(round(np.mean(iou_list_gradCPP_cum), 3))


if __name__ == '__main__':
    current_location = os.getcwd()
    pathOutput = os.path.join(current_location, "heatmap_output")
    if not os.path.exists(pathOutput):
        os.makedirs(pathOutput)

    args = parser.parse_args()


    print("#########################################################################")
    print("#########################################################################")
    print("                             LSE pooling")
    print("#########################################################################")
    print("#########################################################################")
    pathModel = os.path.join(current_location, "savedModels", "compute",
                             "ResNet18_True_LSE_IMG_SIZE_512_num_class_8_best_model.pth")
    h = HeatmapGenerator(pathModel, save_plots=False, args=args)

    h.generate(plot=False, factor=1.7666, threshold_high=0.7666)

    args.global_pool = 'LSE'
    factors = np.linspace(1.4, 2.2, 7)
    threshold_highs = np.linspace(0.5, 0.9, 7)

    for x in factors:
        for y in threshold_highs:
            h.generate(plot=False, factor=x, threshold_high=y)

    plt.figure()
    x = factors
    y = threshold_highs
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.array(h.iou_GradCAMCPP).reshape(x.shape[0], y.shape[0])
    plt.xlabel('alpha value')
    plt.ylabel('threshold')
    im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    plt.colorbar(im)  # adding the colobar on the right
    x_max_index = x[np.array(h.iou_GradCAMCPP).argmax() // y.shape[0]]
    y_max_index = y[np.array(h.iou_GradCAMCPP).argmax() % y.shape[0]]
    plt.title("LSE GradCAM++ MAX IOU is : " + str(np.array(h.iou_GradCAMCPP).max()))
    plt.show()
    print("Threshold " + str(x_max_index) + " and alpha value " + str(y_max_index))

    plt.figure()
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.array(h.iou_GradCAM).reshape(x.shape[0], y.shape[0])
    plt.xlabel('alpha value')
    plt.ylabel('threshold')
    im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    plt.colorbar(im)  # adding the colobar on the right
    x_max_index = x[np.array(h.iou_GradCAM).argmax() // y.shape[0]]
    y_max_index = y[np.array(h.iou_GradCAM).argmax() % y.shape[0]]
    plt.title("LSE GradCAM MAX IOU is : " + str(np.array(h.iou_GradCAM).max()))
    plt.show()
    print("Threshold " + str(x_max_index) + " and at alpha value " + str(y_max_index))

    plt.figure()
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.array(h.iou_CAM).reshape(x.shape[0], y.shape[0])
    plt.xlabel('alpha value')
    plt.ylabel('threshold')
    im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    plt.colorbar(im)  # adding the colobar on the right
    x_max_index = x[np.array(h.iou_CAM).argmax() // y.shape[0]]
    y_max_index = y[np.array(h.iou_CAM).argmax() % y.shape[0]]
    plt.title("LSE CAM MAX IOU is : " + str(np.array(h.iou_CAM).max()))
    plt.show()
    print("Threshold " + str(x_max_index) + " and at alpha value " + str(y_max_index))

    print("#########################################################################")
    print("#########################################################################")
    print("                               MAX Pooling")
    print("#########################################################################")
    print("#########################################################################")
    args.global_pool = 'MAX'
    h = HeatmapGenerator(pathModel, save_plots=False, args=args)
    pathModel = os.path.join(current_location, "savedModels", "compute",
                             "densenet121_True_MAX_IMG_SIZE_224_num_class_8_best_model.pth")
    h.generate(plot=True, factor=1.9, threshold_high=0.75)
    for x in factors:
        for y in threshold_highs:
            h.generate(plot=False, factor=x, threshold_high=y)

    plt.figure()
    x = factors
    y = threshold_highs
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.array(h.iou_GradCAMCPP).reshape(x.shape[0], y.shape[0])
    plt.xlabel('alpha value')
    plt.ylabel('threshold')
    im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    plt.colorbar(im)  # adding the colobar on the right
    x_max_index = x[np.array(h.iou_GradCAMCPP).argmax() // y.shape[0]]
    y_max_index = y[np.array(h.iou_GradCAMCPP).argmax() % y.shape[0]]
    plt.title("MAX GradCAM++ MAX IOU is : " + str(np.array(h.iou_GradCAMCPP).max()))
    plt.show()
    print("Threshold " + str(x_max_index) + " and alpha value " + str(y_max_index))

    plt.figure()
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.array(h.iou_GradCAM).reshape(x.shape[0], y.shape[0])
    plt.xlabel('alpha value')
    plt.ylabel('threshold')
    im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    plt.colorbar(im)  # adding the colobar on the right
    x_max_index = x[np.array(h.iou_GradCAM).argmax() // y.shape[0]]
    y_max_index = y[np.array(h.iou_GradCAM).argmax() % y.shape[0]]
    plt.title("MAX GradCAM MAX IOU is : " + str(np.array(h.iou_GradCAM).max()))
    plt.show()
    print("Threshold " + str(x_max_index) + " and at alpha value " + str(y_max_index))

    plt.figure()
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.array(h.iou_CAM).reshape(x.shape[0], y.shape[0])
    plt.xlabel('alpha value')
    plt.ylabel('threshold')
    im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    plt.colorbar(im)  # adding the colobar on the right
    x_max_index = x[np.array(h.iou_CAM).argmax() // y.shape[0]]
    y_max_index = y[np.array(h.iou_CAM).argmax() % y.shape[0]]
    plt.title("MAX CAM MAX IOU is : " + str(np.array(h.iou_CAM).max()))
    plt.show()
    print("Threshold " + str(x_max_index) + " and at alpha value " + str(y_max_index))

    print("#########################################################################")
    print("#########################################################################")
    print("                           AVG Pooling")
    print("#########################################################################")
    print("#########################################################################")
    args.global_pool = 'AVG'
    pathModel = os.path.join(current_location, "savedModels", "compute",
                             "densenet121_True_AVG_IMG_SIZE_224_num_class_8_best_model.pth")
    h = HeatmapGenerator(pathModel, save_plots=False, args=args)

    for x in factors:
        for y in threshold_highs:
            h.generate(plot=False, factor=x, threshold_high=y)

    plt.figure()
    x = factors
    y = threshold_highs
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.array(h.iou_GradCAMCPP).reshape(x.shape[0], y.shape[0])
    plt.xlabel('alpha value')
    plt.ylabel('threshold')
    im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    plt.colorbar(im)  # adding the colobar on the right
    x_max_index = x[np.array(h.iou_GradCAMCPP).argmax() // y.shape[0]]
    y_max_index = y[np.array(h.iou_GradCAMCPP).argmax() % y.shape[0]]
    plt.title("AVG GradCAM++ MAX IOU is : " + str(np.array(h.iou_GradCAMCPP).max()))
    plt.show()
    print("Threshold " + str(x_max_index) + " and alpha value " + str(y_max_index))

    plt.figure()
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.array(h.iou_GradCAM).reshape(x.shape[0], y.shape[0])
    plt.xlabel('alpha value')
    plt.ylabel('threshold')
    im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    plt.colorbar(im)  # adding the colobar on the right
    x_max_index = x[np.array(h.iou_GradCAM).argmax() // y.shape[0]]
    y_max_index = y[np.array(h.iou_GradCAM).argmax() % y.shape[0]]
    plt.title("AVG GradCAM MAX IOU is : " + str(np.array(h.iou_GradCAM).max()))
    plt.show()
    print("Threshold " + str(x_max_index) + " and at alpha value " + str(y_max_index))

    plt.figure()
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.array(h.iou_CAM).reshape(x.shape[0], y.shape[0])
    plt.xlabel('alpha value')
    plt.ylabel('threshold')
    im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    plt.colorbar(im)  # adding the colobar on the right
    x_max_index = x[np.array(h.iou_CAM).argmax() // y.shape[0]]
    y_max_index = y[np.array(h.iou_CAM).argmax() % y.shape[0]]
    plt.title("AVG CAM MAX IOU is : " + str(np.array(h.iou_CAM).max()))
    plt.show()
    print("Threshold " + str(x_max_index) + " and at alpha value " + str(y_max_index))



    #PCAM
    # pathModel = os.path.join(current_location, "savedModels",
    #                          "densenet121_True_PCAM_IMG_SIZE_224_num_class_8_best_model.pth")
    # h = HeatmapGenerator(pathModel, save_plots=False, args=args)
    # h.generatePCAM(plot=False, factor=1.95, threshold_high=0.7)
    # factors = np.linspace(1.4, 2.2, 7)
    # threshold_highs = np.linspace(0.5, 0.9, 7)
    # for x in factors:
    #     for y in threshold_highs:
    #         h.generatePCAM(plot=False, factor=x, threshold_high=y)


