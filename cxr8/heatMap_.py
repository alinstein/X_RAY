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
from X_RAY.cxr8.utils import visualize_cam


class HeatmapGenerator:

    # ---- Initialize heatmap generator
    # ---- pathModel - path to the trained densenet model
    # ---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    # ---- nnClassCount - class count, 14 for chxray-14

    def __init__(self, pathModel, save_plots):
        # Class names
        self.class_names = ['Atelectasis', 'Cardiomegaly',
                            'Effusion', 'Infiltrate',
                            'Mass', 'Nodule',
                            'Pneumonia', 'Pneumothorax']

        # ---- Initialize the network
        self.save_plots = save_plots
        self.save_location = os.path.join(os.getcwd(), "heatmap_output")
        self.args = parser.parse_args()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_GPU = torch.cuda.device_count()
        args = parser.parse_args()
        model = select_model(args)
        model = model.to(self.device)

        modelCheckpoint = torch.load(pathModel)
        if num_GPU > 1:
            model.module.load_state_dict(modelCheckpoint['model_state_dict'])
        else:
            model.load_state_dict(modelCheckpoint['model_state_dict'])

        self.model = model
        self.model.eval()

        if args.backbone == "densenet121":
        #     model_dict = dict(type='densenet',
        #                       layer_name='densenet121_features_norm5',
        #                       arch=self.model,
        #                       input_size=(args.img_size, args.img_size)
        #                       )
        # elif args.backbone == "densenet121":
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

        self.GradCAM = Grad_CAM(model_dict)
        self.GradCAMCPP = Grad_CAMpp(model_dict)

        # ---- Initialize the weights
        if args.backbone == "densenet121":
            self.weights = list(self.model.densenet121.classifier._modules['0'].parameters())[-2]
        elif args.backbone == "densenet121_AVG":
            self.weights = list(self.model.FF.parameters())[-2].squeeze()
        elif args.backbone == "ResNet18":
            self.weights = list(model.img_model.fc[1].parameters())[-2].squeeze()


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
    def get_iou(self, pred_box, gt_box):
        """
        pred_box : the coordinate for predict bounding box
        gt_box :   the coordinate for ground truth bounding box
        return :   the iou score
        the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
        the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
        """
        x_1 = pred_box[0]
        y_1 = pred_box[1]
        x_2 = pred_box[0] + pred_box[2]
        y_2 = pred_box[1] + pred_box[3]
        pred_box[0] = x_1
        pred_box[1] = y_1
        pred_box[2] = x_2
        pred_box[3] = y_2

        area_pred_box = abs(x_1-x_2)*abs(y_1-y_2)

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

        return iou , iobb


    def large_to_small(self, x1, y1, w1, h1,
                       cropped=True, newsize=512):
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

    def boxIouPlot(self, npHeatmap,  imgOriginal, bbox, threshold_high, factor, plot, label_name, method, name):

        cam = npHeatmap - np.min(npHeatmap)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, (self.args.img_size, self.args.img_size))
        bbox = self.large_to_small(*bbox.numpy()[0], cropped=True, newsize=self.args.img_size)

        thresh_low = cam.mean() * factor
        if thresh_low > threshold_high:
            thresh_low = threshold_high
        thresh_low = float(thresh_low)

        if plot:
            plt.imshow(cam)
            ax = plt.gca()
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.title("Method : "+ method+ "   Disease : " + str(label_name))
            if self.save_plots:
                plt.savefig(os.path.join(self.save_location, (name[0].split(".")[0] + " __" + method + "_" + label_name)))
            plt.show()
        _, thresh = cv2.threshold(cam, thresh_low, 1, cv2.THRESH_BINARY)
        if plot:
            plt.figure()
            plt.imshow(thresh.astype(np.uint8))
            ax = plt.gca()
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.title("Method : "+ method+ "   Disease : " + str(label_name))
        img, contours, hierarchy = cv2.findContours(
            (thresh.astype(np.uint8)),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # The first order of the contours
        max_iou = 0
        cum_iou = 0
        max_iobb = 0
        cum_iobb = 0
        for c_0 in contours:

            # Get the 4 points of the bounding rectangle
            x, y, w, h = cv2.boundingRect(c_0)
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
        if plot:

            if self.save_plots:
                plt.savefig(os.path.join(self.save_location, (name[0].split(".")[0] + " _Bbox_"
                                                              + method + "_" + label_name)))
            plt.show()
            plt.figure()
            heatmap, cam_result = visualize_cam(cam, imgOriginal)
            cam_result = cam_result.squeeze().permute(1, 2, 0)
            plt.imshow(cam_result.detach().cpu().numpy())
            if self.save_plots:
                plt.savefig(os.path.join(self.save_location, (name[0].split(".")[0]+" _XRay-merge_"
                                                              + method + "_" + label_name)))
            plt.show()
        if not contours:
            return 0, 0, 0, 0
        else:
            return max_iou, cum_iou, cum_iobb, max_iobb


    def generate(self, plot=False,factor=1.95, threshold_high=0.6):

        # ---- Load image, transform, convert
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
                    output = self.model.densenet121.features(imageData)
                    output = F.relu(output, inplace=True)
                elif self.args.backbone == "densenet121_AVG":
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
                # thresh_low = mask.mean()  * factor
                # if thresh_low > threshold_high:
                #     thresh_low = threshold_high
                # thresh_low = float(thresh_low)
                iou, cum_iou, cum_iobb, max_iobb = self.boxIouPlot(mask.detach().squeeze().cpu().numpy(), imgOriginal,
                                                    bbox, threshold_high, factor, plot, label_name[0], "GradCAM", name)
                iou_list_grad.append(iou)
                iou_dic_grad[label_name[0]].append(iou)
                iou_list_grad_cum.append(cum_iou)
                iou_dic_grad_cum[label_name[0]].append(cum_iou)
                iobb_list_grad_cum.append(cum_iobb)
                iobb_dic_grad_cum[label_name[0]].append(cum_iobb)


                # GradCAMCPP method
                imageData_temp = Variable(imageData, requires_grad=True)
                mask, logit = self.GradCAMCPP(imageData_temp, class_idx=label)
                # thresh_low = mask.mean() * factor
                # if thresh_low > threshold_high:
                #     thresh_low = threshold_high
                # thresh_low = float(thresh_low)
                iou, cum_iou, cum_iobb, max_iobb = self.boxIouPlot(mask.detach().squeeze().cpu().numpy(), imgOriginal,
                                                bbox, threshold_high, factor, plot, label_name[0], "GradCAM++", name)
                iou_list_gradCPP.append(iou)
                iou_dic_gradCPP[label_name[0]].append(iou)
                iou_list_gradCPP_cum.append(cum_iou)
                iou_dic_gradCPP_cum[label_name[0]].append(cum_iou)
                iobb_list_gradCPP_cum.append(cum_iobb)
                iobb_dic_gradCPP_cum[label_name[0]].append(cum_iobb)

                # CAM method
                # ---- Generate heatmap
                #TODO check the implemenation of CAM (weights)
                heatmap = None
                weights = self.weights[label]
                for i in range(0, len(weights)):
                    map = output[0, i, :, :]
                    if i == 0:
                        heatmap = weights[i] * map
                    else:
                        heatmap += weights[i] * map
                    npHeatmap = heatmap.cpu().data.numpy()

                # thresh_low = npHeatmap.mean() * factor
                # if thresh_low > threshold_high:
                #     thresh_low = threshold_high
                iou, cum_iou, cum_iobb, max_iobb = self.boxIouPlot(npHeatmap, imgOriginal, bbox,
                                                    threshold_high, factor, plot, label_name[0], "CAM", name)
                iou_list.append(iou)
                iou_dic[label_name[0]].append(iou)
                iou_list_cum.append(cum_iou)
                iou_dic_cum[label_name[0]].append(cum_iou)
                iobb_list_cum.append(cum_iobb)
                iobb_dic_cum[label_name[0]].append(cum_iobb)

                # cam = npHeatmap - np.min(npHeatmap)
                # cam = cam / np.max(cam)
                # cam = cv2.resize(cam, (self.args.img_size, self.args.img_size))
                # bbox = self.large_to_small(*bbox.numpy()[0], cropped=True, newsize=self.args.img_size)
                #
                # if plot:
                #     plt.imshow(cam)
                #     ax = plt.gca()
                #     rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                #                              linewidth=1, edgecolor='r', facecolor='none')
                #     ax.add_patch(rect)
                #     plt.show()
                # #threshold = (cam.mean() + 0.5 * cam.std()) * factor
                # threshold = cam.mean() * factor
                # if threshold > threshold_high:
                #     threshold = threshold_high
                # threshold = float(threshold)
                # _ , thresh = cv2.threshold(cam, threshold, 1, cv2.THRESH_BINARY)
                # if plot:
                #     plt.figure()
                #     plt.imshow(thresh.astype(np.uint8))
                #     ax = plt.gca()
                #     rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                #                              linewidth=1, edgecolor='r', facecolor='none')
                #     ax.add_patch(rect)
                # img, contours, hierarchy = cv2.findContours(
                #     (thresh.astype(np.uint8)),
                #     cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # # The first order of the contours
                # max_iou = 0
                # cum_iou = 0
                # cum_iobb = 0
                # max_iobb = 0
                # for c_0 in contours:
                #
                #     # Get the 4 points of the bounding rectangle
                #     x, y, w, h = cv2.boundingRect(c_0)
                #     Pred_bbox = [x, y, w, h]
                #     # Draw a straight rectangle with the points
                #     if plot:
                #         rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g',
                #                                  facecolor='none')
                #         ax.add_patch(rect)
                #
                #     iou, iobb = self.get_iou(list(bbox), Pred_bbox)
                #     cum_iou = cum_iou + iou
                #     cum_iobb = cum_iobb + iobb
                #     if max_iou < iou:
                #         max_iou = iou
                #     if max_iobb < iobb:
                #         max_iobb = iobb
                # if cum_iobb > 1:
                #     cum_iobb = 1
                #
                # if plot:
                #     plt.show()
                #
                # iou_list.append(max_iou)
                # iou_dic[label_name[0]].append(iou)
                # iou_list_cum.append(cum_iou)
                # iou_dic_cum[label_name[0]].append(cum_iou)
                #
                # iobb_list_cum.append(cum_iobb)
                # iobb_dic_cum[label_name[0]].append(cum_iobb)

        print("\n\n\n")
        print("_______________________________________________________")
        print("Factor is :", factor, " & threshold is :", threshold_high)
        print("\n\n")
        print("______________________________________")
        print("______________CAM_____________________")
        print("______________________________________")
        print("Mean of iou", round(np.mean(iou_list), 3))
        for name in iou_dic.keys():
            print(name, " : ",  round(np.mean(iou_dic[name]),3))
        print("\n")
        print("Cumulative IOU")
        print(round(np.mean(iou_list_cum), 3))
        for name in iou_dic_cum.keys():
            print(name, " : ",  round(np.mean(iou_dic_cum[name]),3))
        print("\n")
        print("Mean of ioBB", round(np.mean(iobb_list_cum), 3))
        print("\n")
        print("Greater than 0.25 IOU")
        print(np.mean(np.array(iou_list_cum) >= 0.25))
        for name in iou_dic_cum.keys():
            print(name, np.mean(np.array(iou_dic_cum[name]) >= 0.25) )
        print("\n\n")

        print("______________________________________")
        print("_____________GradCAM__________________")
        print("______________________________________")
        print("Mean of iou",round(np.mean(iou_list_grad),3))
        for name in iou_dic_grad.keys():
            print(name, " : ", round(np.mean(iou_dic_grad[name]),3))
        print("\n")
        print("Cumulative IOU")
        print(round(np.mean(iou_list_grad_cum), 3))
        for name in iou_dic_grad_cum.keys():
            print(name, " : ", round(np.mean(iou_dic_grad_cum[name]),3))
        print("\n")
        print("Mean of ioBB", round(np.mean(iobb_list_grad_cum), 3))
        print("\n")
        print("Greater than 0.25 IOU")
        print(np.mean(np.array(iou_list_grad_cum) >= 0.25))
        for name in iou_dic_grad_cum.keys():
            print(name, np.mean(np.array(iou_dic_grad_cum[name]) >= 0.25) )
        print("\n\n")

        print("______________________________________")
        print("____________GradCAM++_________________")
        print("______________________________________")
        print("Mean of iou",round(np.mean(iou_list_gradCPP),3))
        for name in iou_dic_gradCPP.keys():
            print(name, " : ", round(np.mean(iou_dic_gradCPP[name]),3))
        print("Cumulative IOU")
        print(round(np.mean(iou_list_gradCPP_cum), 3))
        for name in iou_dic_gradCPP_cum.keys():
            print(name, " : ", round(np.mean(iou_dic_gradCPP_cum[name]), 3))
        print("\n")
        print("Mean of ioBB", round(np.mean(iobb_list_gradCPP_cum), 3))
        print("\n")
        print("Greater than 0.25 IOU")
        print(np.mean(np.array(iou_list_gradCPP_cum) >= 0.25))
        for name in iou_dic_gradCPP_cum.keys():
            print(name, np.mean(np.array(iou_dic_gradCPP_cum[name]) >= 0.25) )

        self.iou_CAM.append(round(np.mean(iou_list_cum), 3))
        self.iou_GradCAM.append(round(np.mean(iou_list_grad_cum), 3))
        self.iou_GradCAMCPP.append(round(np.mean(iou_list_gradCPP_cum), 3))


if __name__ == '__main__':
    current_location = os.getcwd()
    pathOutput = os.path.join(current_location, "heatmap_output")
    if not os.path.exists(pathOutput):
        os.makedirs(pathOutput)
    pathModel = os.path.join(current_location, "savedModels",
                             "ResNet18_MAX_IMG_SIZE_224num_class_8_best_model.pth")
    h = HeatmapGenerator(pathModel, save_plots=False)
    h.generate(plot=False, factor=1.9, threshold_high=0.75)
    # factors = np.linspace(0.8, 2.2, 5)
    # threshold_highs = np.arange(0.2, 0.91, 5)
    #
    # for x in factors:
    #     for y in threshold_highs:
    #         h.generate(plot=False, factor=x, threshold_high=y)
    #
    # plt.figure()
    # x = factors
    # y = threshold_highs
    # X, Y = np.meshgrid(x, y)  # grid of point
    # Z = np.array(h.iou_GradCAMCPP).reshape(x.shape[0], y.shape[0])
    # plt.xlabel('alpha value')
    # plt.ylabel('threshold')
    # im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    # plt.colorbar(im)  # adding the colobar on the right
    # x_max_index = x[np.array(h.iou_GradCAMCPP).argmax() // y.shape[0]]
    # y_max_index = y[np.array(h.iou_GradCAMCPP).argmax() % y.shape[0]]
    # plt.title("GradCAM++ MAX IOU is : " + str(np.array(h.iou_GradCAMCPP).max()))
    # plt.show()
    # print("Threshold " + str(x_max_index) + " and alpha value " + str(y_max_index))
    #
    # plt.figure()
    # X, Y = np.meshgrid(x, y)  # grid of point
    # Z = np.array(h.iou_GradCAM).reshape(x.shape[0], y.shape[0])
    # plt.xlabel('alpha value')
    # plt.ylabel('threshold')
    # im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    # plt.colorbar(im)  # adding the colobar on the right
    # x_max_index = x[np.array(h.iou_GradCAM).argmax() // y.shape[0]]
    # y_max_index = y[np.array(h.iou_GradCAM).argmax() % y.shape[0]]
    # plt.title("GradCAM MAX IOU is : " + str(np.array(h.iou_GradCAM).max()))
    # plt.show()
    # print("Threshold " + str(x_max_index) + " and at alpha value " + str(y_max_index))
    #
    # plt.figure()
    # X, Y = np.meshgrid(x, y)  # grid of point
    # Z = np.array(h.iou_CAM).reshape(x.shape[0], y.shape[0])
    # plt.xlabel('alpha value')
    # plt.ylabel('threshold')
    # im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=[y[0], y[-1], x[-1], x[0]])
    # plt.colorbar(im)  # adding the colobar on the right
    # x_max_index = x[np.array(h.iou_CAM).argmax() // y.shape[0]]
    # y_max_index = y[np.array(h.iou_CAM).argmax() % y.shape[0]]
    # plt.title("CAM MAX IOU is : " + str(np.array(h.iou_CAM).max()))
    # plt.show()
    # print("Threshold " + str(x_max_index) + " and at alpha value " + str(y_max_index))


