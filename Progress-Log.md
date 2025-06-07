# Progress Log

## Attempt no. 1

Achieved **88%** `training accuracy`, and **81%** in `testing accuracy` and `validation accuracy`.

## Attempt no. 2

Tried a bunch of different configuration in the model Architecture to improve the accuracy, but ended up either decreasing it or devastating it.

Tried using `BatchNormalization()`, removing `Data Augmentation`, increasing `IMG_SIZE`, and also increasing the number of `Conv2D` and `MaxPooling2D` layers.

Finally, after a lot of different configurations, what worked considerablly well is just adding a `Dropout` layer.
It resulted in a `training accuracy` of **79%**, `validation accuracy` of **79%**, and a `testing accuracy` of **77%**.
Although the accuracy is slightly lower, but over-fitting is slightly reduced.

## Attempt no. 3

Just a few changes in the model, changed the position of `Dropout` layer and added another one, added another set of `Conv2D` and `MaxPooling2D` layers.
Also increase the number of epochs from **10** to **20**.
Achieved a `training accuracy` of **90%**, `validation accuracy` of **83%**, and a `testing accuracy` of **82%**.

I guess, will have to try `Transfer Learning` in the next attempt, considering the small size of the dataset.

## Attempt no. 4

Went with `MobileNetV2` for Transfer Learning. 

Made a slight mistake of forgetting to include one line in the first attempt,
``` python
base_model.trainable = False
```
And it lead to some *Not so Great* Results.

Resolved this problem, and trained the top layers for `15 Epochs`.
Achieved **95.7%** `Training Accuracy`, **95.2%** `Validation Accuracy`, and **96%** `Testing Accuracy`.

## Attempt at a Binary Classifier

Implemented a Convolutional Binary Classifier for detecting if a given image is of a **Lemon Leaf** or not.
Gathered a makeshift dataset by combining images from `PlantVillage` and `Natural Images` datasets to consist of images for *Not Lemon Leaves* and images from the `Lemon Leaf Disease Dataset` to consist of images for *Lemon Leaves*.

Also went with Transfer Learning using the same `MobileNetV2`, and achieved a `training accuracy` of **98.9%**, `validation accuracy` of **100%**, and a `testing accuracy` of **99.7%**.

So the idea is basically to first input a image from the `Binary Classifier` to confirm if the given image is of a *Lemon Leaf*, if affirmative, it would be given as an input to `Multi-Class Classifier` that is just been implemented.