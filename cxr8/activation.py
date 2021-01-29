import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import os
from collections import OrderedDict

from X_RAY.cxr8.gradcam import Grad_CAM, Grad_CAMpp
from X_RAY.cxr8.utils import visualize_cam
from dataset import CXRDataset
from train import parser, select_model

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
args = parser.parse_args()
current_location = os.getcwd()
data_root_dir = os.path.join(current_location, 'dataset')


processed = True

# with open(test_txt_path, "r") as f:
#     test_list = [i.strip() for i in f.readlines()]

# print("number of test examples:", len(test_list))

# if processed is True, loading an intermediate array is much quicker
# if processed == False:
#     test_X = []
#     print("load and transform image")
#     for i in range(len(test_list)):
#         image_path = os.path.join(img_folder_path, test_list[i])
#         img = scipy.misc.imread(image_path)  # imageio.imread
#         if img.shape != (1024, 1024):
#             img = img[:, :, 0]
#         img_resized = skimage.transform.resize(img, (256, 256))
#         test_X.append((np.array(img_resized)).reshape(256, 256, 1))
#         if i % 100 == 0:
#             print(i)
#     test_X = np.array(test_X)
#     np.save("/home/ubuntu/project/data/postproc/test_bbox_X_small.npy", test_X)
# else:
#     test_X = np.load("/home/ubuntu/project/data/postproc/test_bbox_X_small.npy")


trans = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
datasets = CXRDataset(data_root_dir, dataset_type='box', Num_classes=args.num_classes,
                      img_size=args.img_size, transform=trans)
dataloaders = DataLoader(datasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


nnClassCount = args.num_classes
# ---- Initialize the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_GPU = torch.cuda.device_count()
args = parser.parse_args()
model = select_model(args)

if torch.cuda.is_available():
    model = (model).cuda()
else:
    model = model

current_location = os.getcwd()
pathModel = os.path.join(current_location,  "savedModels", "densenet121_LSE_IMG_SIZE_224num_class_8_best_model.pth")
modelCheckpoint = torch.load(pathModel)
if num_GPU > 1:
    model.module.load_state_dict(modelCheckpoint['model_state_dict'])
else:
    model.load_state_dict(modelCheckpoint['model_state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

thresholds = np.load("thresholds.npy")
print("activate threshold", thresholds)

print("generate heatmap ..........")


# This gcam function is from T.H. Tang's code. I did not end up using this for my final project.
# ======= Grad CAM Function =========
class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
        #         self.probs = F.softmax(self.preds)[0]
        #         self.prob, self.idx = self.preds[0].data.sort(0, True)
        return self.preds.cpu().data.numpy()

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():  # search the ordered dict
            for module in self.model.named_modules():  # 437 tuples (name, layer type)
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]  # (7,7)
        return nn.AvgPool2d(self.map_size)(grads)  # [1, 32, 1, 1]p

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)  # [1, 32, 7, 7])
        grads = self._find(self.all_grads, target_layer)  # [1, 32, 7, 7,]
        weights = self._compute_grad_weights(grads)  # [1, 32, 1, 1]
        gcam = torch.FloatTensor(self.map_size).zero_()  # (7,7)
        for fmap, weight in zip(fmaps[0], weights[0]):  # loop over 32m
            gcam += fmap * weight.data

        gcam = F.relu(Variable(gcam))  # only positives

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))  # 224, 224

        return gcam

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


# ======== Create heatmap ===========

gcam_outputs = []  # (224,224) activation maps forp each class activation, from T.H. Tang

# from Grad-CAM, by WonKwang Lee ------ these did not work
gradcam_masks = []  # class activation map
gradcam_heatmaps = []  # class activation map, RGB
gradcam_results = []  # heatmap over image

# from Grad-CAM++, by WonKwang Lee
gradcampp_masks = []  # class activation map
gradcampp_heatmaps = []  # class activation map, RGB
gradcampp_results = []  # heatmap over image

image_id = []  # index of activation map example
output_class = []  # class of activation mapmap

gcam = GradCAM(model=model, cuda=True)  # T.H.Tang
model.eval()
model_dict = dict(type='densenet', arch=model, layer_name='densenet121_features_norm5', input_size=(224, 224))
gradcam = Grad_CAM(model_dict)  # WonKwang Lee
gradcampp = Grad_CAMpp(model_dict)  # WonKwang Lee

for index, (image, label, bbox, name) in enumerate(tqdm(dataloaders)):
    input_img = image.to(device)
    probs = gcam.forward(input_img)  # input test example through gcam model, should get probs

    activate_classes = np.where((probs > thresholds)[0] == True)[0]  # get the activated class
    for activate_class in activate_classes:  # for all classes that are above threshold
        gcam.backward(idx=activate_class)  # backprop
        gcam_output = gcam.generate(target_layer="densenet121.features.norm5")
        mask, _ = gradcam(input_img, class_idx=activate_class)
        heatmap, cam_result = visualize_cam(mask, input_img)
        mask = mask.view(224, 224).cpu().numpy()

        maskpp, _ = gradcampp(input_img, class_idx=activate_class)
        heatmappp, cam_resultpp = visualize_cam(maskpp, input_img)
        maskpp = maskpp.view(224, 224).cpu().numpy()

        gcam_outputs.append(gcam_output)
        gradcam_masks.append(mask)
        gradcam_heatmaps.append(heatmap.numpy())
        gradcam_results.append(cam_result.detach().numpy())

        gradcampp_masks.append(maskpp)
        gradcampp_heatmaps.append(heatmappp.numpy())
        gradcampp_results.append(cam_resultpp.detach().numpy())
        image_id.append(index)
        output_class.append(activate_class)

    # print("test ", str(index), " finished")

print("done processing CAM")
print("saving outputs")  # save outputs for speed
np.save(os.path.join(current_location,"activation_maps", "gcam_output.npy"), np.stack(gcam_outputs))
np.save(os.path.join(current_location,"activation_maps","gradcam_masks.npy"), np.stack(gradcam_masks))
np.save(os.path.join(current_location,"activation_maps","gradcam_heatmaps.npy"), np.stack(gradcam_heatmaps))
np.save(os.path.join(current_location,"activation_maps","gradcam_results.npy"), np.stack(gradcam_results))
np.save(os.path.join(current_location,"activation_maps","gradcampp_masks.npy"), np.stack(gradcampp_masks))
np.save(os.path.join(current_location,"activation_maps","gradcampp_heatmaps.npy"), np.stack(gradcampp_heatmaps))
np.save(os.path.join(current_location,"activation_maps","gradcampp_results.npy"), np.stack(gradcampp_results))
np.save(os.path.join(current_location,"activation_maps","image_id.npy"), np.asarray(image_id))
np.save(os.path.join(current_location,"activation_maps","output_class.npy"), np.asarray(output_class))

print("heatmap output done")
# print("total number of heatmap: ", len(heatmap_output))