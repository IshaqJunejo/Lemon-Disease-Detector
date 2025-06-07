# Lemon Disease Detector

## Plan

So the plan is to create 2 `Convolutional Neural Networks`, a `Binary Classifier` and a `Multi-Class Classifier`.

The `Binary Classifier` should be able to tell apart if a given image is of a *Lemon Leaf* or not. If the image is of a *Lemon Leaf*, it would be given as an input to the Next Model.

The `Multi-Class Classifier` should be able to predict if the *Lemon Leaf* is Healthy or has any of the given diseases.
- Anthracnose
- Bacterial Blight
- Citrus Canker
- Curl Virus
- Deficiency Leaf
- Dry Leaf
- Sooty Mould
- Spider Mites

The plan is mostly executed, one thing that remains is to use the combination of both these models in a basic web-app to test the architecture in the real-world.

### Dataset

- [Lemon Leaf Disease Dataset](https://www.kaggle.com/datasets/mahmoudshaheen1134/lemon-leaf-disease-dataset-lldd) on Kaggle
- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) on Kaggle
- [Natural Images Dataset](https://www.kaggle.com/datasets/prasunroy/natural-images) on Kaggle

Images from `PlantVillage` and `Natural Images` dataset were used as *Not Lemon Leaves* for training and testing the `Binary Classification Model`.