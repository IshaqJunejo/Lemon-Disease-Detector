# Progress Log

## First Attempt

Achieved **88%** `training accuracy`, and **81%** in `testing accuracy` and `validation accuracy`.

## Second Attempt

Tried a bunch of different configuration in the model Architecture to improve the accuracy, but ended up either decreasing it or devastating it.

Tried using `BatchNormalization()`, removing `Data Augmentation`, increasing `IMG_SIZE`, and also increasing the number of `Conv2D` and `MaxPooling2D` layers.

Finally, after a lot of different configurations, what worked considerablly well is just adding a `Dropout` layer.
It resulted in a `training accuracy` of **79%**, `validation accuracy` of **79%**, and a `testing accuracy` of **77%**.
Although the accuracy is slightly lower, but over-fitting is slightly reduced.