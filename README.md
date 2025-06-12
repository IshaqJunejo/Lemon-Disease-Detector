# Lemon Disease Detector

A Bi-model Deep Learning Architecture to detect the disease of Lemon Leaves using `Convolutional Neural Networks`.

It can categorize Lemon Leaves into Healthy Leaves, and into these Diseases,
- Anthracnose
- Bacterial Blight
- Citrus Canker
- Curl Virus
- Deficiency Leaf
- Dry Leaf
- Sooty Mould
- Spider Mites

## Architecture

Architecture is based on 2 `Convolutional Neural Networks`.

First one is a `Binary Classification Model` that can tell if a given image is a *Lemon Leaf* or *Not*. The Image is passed to second model only if it is a *Lemon Leaf*.

Second model is a `Multi-Class Classifier` that tells the likelihood of the given image of Lemon Leaf having each *disease* and of being *healthy*.

Both models are based of `MobileNetV2` for Transfer Learning, as there was limited number of images to train a model from scratch.

### Dataset

- [Lemon Leaf Disease Dataset](https://www.kaggle.com/datasets/mahmoudshaheen1134/lemon-leaf-disease-dataset-lldd) on Kaggle
- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) on Kaggle
- [Natural Images Dataset](https://www.kaggle.com/datasets/prasunroy/natural-images) on Kaggle

Images from `PlantVillage` and `Natural Images` dataset were used as *Not Lemon Leaves* for training and testing the `Binary Classification Model`.