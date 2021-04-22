# Weakly supervised Classification and Localization of Chest X-ray images.

![Results](https://github.com/alinstein/X_RAY/blob/main/pictures/sample.png)


This project contains two task :
* Detect common thoracic diseases from chest X-ray images. 
* Localize the plausible disease location in X-ray image using CNN model trained on images labels only.

This implementation can used with two chest X-ray image datasets. 
* [NIH Chest X-ray dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community).
* [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)

Following are implemented in this repository:
* Module to train, evaluate the DCNN model on NIH Chest X-ray dataset.
* Can use following CNN : ResNet, DenseNet, Efficient, [PCAM](https://arxiv.org/abs/2005.14480) with DenseNet as backbone.
* Weakly supervised localization methods: Class Activation Map, GradCAM, GradCAM++ to generate a heatmap 
  of plausible disease location and bounding boxes are marked around the disease location.

### Getting Started
1. Install  packages

    
    pip install -r requirements.txt
2. Download the datasets from the link provided above.

3. For NIH dataset extract the dataset and move all images into directory `dataset/images`  and train, val and test CSV files to `dataset`.
   And for NIH dataset before training, it's recommended to reduce the size of image from 1024x1024 to 512x512 or 256x256 depending upon the training image size. 
   This will significantly speed up the training speed.
   
   For ChesXpert dataset, download the dataset and extract all the dataset to `dataset/CheXpert`.
   
4. Change the resolution, CNN architecture, batch size, epochs, pooling layer, attention network etc in config.py.
5. To train and evaluate the multi-label classification model on NIH dataset with eight diseases, run :
    
        python main.py --num_classes=8 --batch_size=32 --multi_label=True --img_size=224 --epochs=20 --dataset=NIH
   
   To train and evaluate the multi-label classification model on NIH dataset with eight diseases, run :
   
        python main.py --num_classes=5 --batch_size=32 --multi_label=True --img_size=224 --epochs=20 --dataset=ChesXpert
   
   To train and evaluate the binary classification (abnormality) model, run :
        
        python main.py --num_classes=1 --batch_size=32 --multi_label=False --img_size=224 --epochs=20


6. To generate a heatmap of the disease, give CNN model's location in main function in heatMap_.py 
   and check the configuration in config.py. Run the following :

         python heatMap_.py

----------------------------------------------
### Reference
[1] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers. ChestX-ray8: [Hospital-scale Chest X-ray Database and Benchmarks on Weakly- Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/pdf/1705.02315.pdf), IEEE CVPR, pp. 3462-3471,2017

[2] J. Irvin, P Rajpurkar, M Ko, Yifan Yu, S Ciurea-Ilcus, C Chute, H Marklund, B Haghgoo, R Ball, K
Shpanskaya, J Seekins, David A. Mong, S S. Halabi, Jesse K. S, Ricky J, David B. L, Curtis P. L, Bhavik N. P, 
Matthew P. L, and Andrew Y. Ng. “CheXpert: A Large Chest Radiograph Dataset With Uncertainty Labels 
and Expert Comparison”. Proceedings of the AAAI Conference on Artificial Intelligence 33 (01):590-97,2019. [Link](https://arxiv.org/pdf/1901.07031.pdf)

[3] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva and A. Torralba, "Learning Deep Features for Discriminative 
Localization," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 
2016, pp. 2921-2929, doi: 10.1109/CVPR.2016.319. [Link](https://arxiv.org/pdf/1512.04150.pdf)

[4] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual 
explanations from deep networks via gradient-based localization," 2017 IEEE International Conference 
on Computer Vision (ICCV), Venice, 2017, pp. 618-626, doi: 10.1109/ICCV.2017.74.
[Link](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_GradCAM_Visual_Explanations_ICCV_2017_paper.pdf)

[5] A. Chattopadhay, A. Sarkar, P. Howlader and V. N. Balasubramanian, "Grad-CAM++: Generalized 
gradient-based visual explanations for deep convolutional networks," 2018 IEEE Winter Conference on 
Applications of Computer Vision (WACV), Lake Tahoe, NV, 2018, pp. 839-847, doi: 10.1
109/WACV.2018.00097. 1710.11063v3.pdf (arxiv.org) [Link](https://arxiv.org/pdf/1710.11063.pdf)
