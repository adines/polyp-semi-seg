# Semi-supervised learning methods for Semantic Segmentation of Polyps

Nowadays, colorectal cancer is one of the most important diseases, and early detection would greatly help improve patient survival. The current methods used by physicians to detect this type of cancer are based on the visual detection of polyps in colonoscopy, a task that can be tackled by means of semantic segmentation methods. However, the amount of data necessary to train a deep learning model for these problems is a barrier for their adoption. In this work, we study the application of different semi-supervised learning techniques to this problem when we have a small amount of annotated data. To carry out this study, we have used the Kvasir-SEG data set, taking only 60 and 120 annotated images and studying the behaviour of the Data Distillation, Model Distillation and Data & Model distillation methods in both cases, using 10 different architectures. The results show that as we increase the number of initially annotated data, the models obtained better results. Furthermore, we can conclude that the Data Distillation method increases the performance of the models between a 25% and 183% . Finally, we can conclude that using only 12% of the annotated data, the results obtained are not very far from those obtained by training the models with the fully annotated dataset. For all these reasons, we conclude that the Data Distillation method is a good tool in semantic segmentation problems when the number of initially annotated images is small, as occurs in many problems in the biomedical field.


## Architectures

We have studied the behaviour of 10 different semantic segmentation networks

|        Architecture        |Backbone | FLOPS (G)  | Parameters (M)           |
|----------------|-------------------------------|-----------------------------|----|
|CGNet| CGNet           |1.4 | 0.5 |
|DeepLabV3+          |ResNet50 | 14.8 | 28.7|
|DenseASPP  | ResNet50 | 14.1 | 29.1|
|FPENet  | FPENet | 0.3 | 0.1 |
|HRNet  | hrnet_w48 | 36.6 | 65.8 |
|LEDNet  | ResNet50 | 2.5 | 2.3 |
|MANet  | ResNet50 | 29.1 | 147.4 |
|OCNet  | ResNet50 | 16.9 | 35.9 |
|PAN  | ResNet50 | 13.6 | 24.3|
|U-Net  | ResNet50 | 48.5 | 13.4 |


## Distillation methods
we will compare three different methods:Data Distillation, Model Distillation and Data & Model Distillation; these methods are based on the notions of self-training and distillation.

### Data Distillation
In the case of Data Distillation, (1) a base model is trained, (2) this model is used to label new images using multiple transformations of the image, and (3) a new model is trained in both, the initial labelled images and the automatically annotated images in (2).

![workflow](assets/DataDitillation.svg)

### Model Distillation
In the case of Model Distillation (1) several models are trained in the initial annotated images, (2) these model are ensembled to label new images, and (3) a new model is trained in both, the initial labelled images and the automatically annotated images in (2).

![workflow](assets/ModelDitillation.svg)

### Data & Model Distillation
Both techniques can also be combined in a technique called Data & Model Distillation.

![workflow](assets/DataModelDitillation.svg)


## Results

We have compared the performance of the 3 distillation methods using only 60 annotated images

Model | Baseline | Data Dist. | Model Dist. | Data-Model Dist.|
--------|-----------|--------------|---------------|------------|
CGNet | 57.63 | 73.34 | 14.9 | 37.42|
DeepLab  | 44.58 | 68.5 | 37.37 | 34.01|
DenseASPP  | 46.94 | 70.45 | 0 | 0|
FPENet  | 46.98 | 61.88 | 39.52 | 44.27|
HRNet  | 65.87 | 66.62 | 14.5 | 3.22|
LEDNet  | 38.47 | 69.05 | 29.8 | 17.75|
MANet  | 56.35 | 70.71 | 0 | 0|
OCNet  | 42.67 | 67.47 | 29.98 | 3.13|
PAN  | 44.72 | 33.83 | 7.49 | 9.12|
U-Net  | 26.08 | 73.95 | 43.27 | 0.8|

Also, we have compared the performance of the 3 distillation methods using only 120 annotated images

Model | Baseline | Data Dist. | Model Dist. | Data-Model Dist.|
--------|-----------|--------------|---------------|------------|
CGNet | 58.51 | 74.89 | 44.11 | 51.88|
DeepLab  | 50.61 | 79.3 | 41.8 | 42.95|
DenseASPP  | 41.23 | 74.83 | 41.79 | 36.06|
FPENet  | 38.63 | 66.97 | 31.55 | 42.38|
HRNet  | 70.11 | 72.14 | 46.13 | 21.12|
LEDNet  | 39.88 | 50.77 | 4.45 | 17.99|
MANet  | 60.18 | 70.74 | 43.07 | 3.6|
OCNet  | 44.94 | 68.95 | 6.72 | 5.4|
PAN  | 59.45 | 37.01 | 0.01 | 9.2|
U-Net  | 76.98 | 79.45 | 36.2 | 34.56|
