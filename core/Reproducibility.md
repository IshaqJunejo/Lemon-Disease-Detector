# Reproducibility Procedure

## Environment Setup

### Clone this Repository

``` bash
git clone https://github.com/IshaqJunejo/Lemon-Disease-Detector.git
cd Lemon-Disease-Detector/core
```

### Virtual Environment
``` bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Directory Structure
```
Lemon-Disease-Detector/
├── api/ 
├── core/
|   └── datasets/
|   └── models/
├── Images/
└── web/
```

## Datasets

- [Lemon Leaf Disease Dataset](https://www.kaggle.com/datasets/mahmoudshaheen1134/lemon-leaf-disease-dataset-lldd) on Kaggle
- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) on Kaggle
- [Natural Images Dataset](https://www.kaggle.com/datasets/prasunroy/natural-images) on Kaggle

Images from `PlantVillage` and `Natural Images` dataset were used as *Not Lemon Leaves* for training and testing the `Binary Classification Model`.

### Directory Structure of Datasets
```
Lemon-Disease-Detector/
├── core/
|   └── datasets/
|       └── Prepared/
|           ├── lemon-leaf-disease-dataset/
|           |   ├── test/
|           |   |   ├── class_1/
|           |   |   ├── class_2/
|           |   |   ├── class_3/
|           |   |   └── ...
|           |   ├── train/
|           |   |   ├── class_1/
|           |   |   ├── class_2/
|           |   |   ├── class_3/
|           |   |   └── ...
|           |   └── validation/
|           |       ├── class_1/
|           |       ├── class_2/
|           |       ├── class_3/
|           |       └── ...
|           └── Lemon-Leaf-or-not/
|               ├── test/
|               |   ├── no/
|               |   └── yes/
|               ├── train/
|               |   ├── no/
|               |   └── yes/
|               └── validate/
|                   ├── no/
|                   └── yes/
```

### Number of Images for Main Model

The `Lemon Leaf Disease Dataset` has a total of 1354 Images in **9 Class** with minimal imbalance.

To train the `Main Model` for disease classification, the dataset has been used with `60-20-20` split in Training, Validation, and Testing. So the model has 812 images to train, 271 images to validate and test each.

### Number of Images for Binary Model

As all the images in `Lemon Leaf Disease Dataset` were to be used as positives for the `Binary Model`, a similar number of negatives were to be picked up from `PlantVillage Dataset` and `Natural Images Dataset`.

It has a total of 2694 Images, with `70-15-15` split in Training, Validation, and Testing. So the model can be trained on 1908 images, validated on 400 images, and tested on 386 images.

## Scripts

What purpose were the scripts in `core/` directory used for,
- `main-model.py` was used as an attempt to train the model from scratch, it is now considered deprecated.
- `saving-mobile-net-v2.py` is used to load the **MobileNetV2** and save it in `.keras` format. It was considered because of possible Internet Outages.
- `transfer-learning.py` is used to train the Main Model using **Transfer Learning** approach, plus image resizing, normalization, and data augmentation are also applied in the script.
- `binary-classifier.py` is used to train the Binary Model using **Transfer Learning** approach, with all the image resizing, normalization, and data augmentation.
- `binary-model-optimization.py` and `main-model-optimization.py` were used in an attempt to perform quantization and save the models in `.tflite` format for optimization, it was done for helping in deploying the models to an Inference API endpoint, which I later decided to not pursue, so now both these scripts are also considered deprecated.
- `testing-model.py` is used to get the Performance Metrics of both the models.