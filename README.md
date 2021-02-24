# Weakly supervised Classification and localization of Chest X-ray images.

![Results](https://github.com/alinstein/X_RAY/blob/main/pictures/sample.png)


Detect common thoracic diseases from chest X-ray images. The dataset used here is [NIH Chest X-ray dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community).
Localize the plausible disease location in X-ray image using CNN model trained on images labels only.

Following are implemented in this repository:
* Module to train, evaluate the DCNN model on NIH Chest X-ray dataset.
* Can use following CNN :ResNet, DenseNet, Efficient, PCAM with DenseNet as backbone.
* Weakly supervised localization methods: Class Activation Map, GradCAM, GradCAM++ to generate a heatmap 
  of plausible disease location and bounding boxes are marked around the disease location.

### Getting Started
1. Install  packages

    
    pip install -r requirements.txt
2. Download the dataset [here]((https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)).
3. Extract the images into directory `dataset/images` and recommend reducing the size of image from 1024x1024 before training unless training in high resolution. 
   This will speed up training significantly.
4. Change the resolution, CNN architecture, batchsize, epochs, pooling layer, attention network etc in config.py.
5. To train and evaluate the model run :


    python train.py
    python eval.py

6. To generate a heatmap of the disease, give CNN model's location in main function in heatMap_.py 
   and check the configuration in config.py.
   

    python heatMap_.py

----------------------------------------------
### Reference
Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers. ChestX-ray8: [Hospital-scale Chest X-ray Database and Benchmarks on Weakly- Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/pdf/1705.02315.pdf), IEEE CVPR, pp. 3462-3471,2017
