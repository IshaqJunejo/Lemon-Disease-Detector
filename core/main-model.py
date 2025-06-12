# This was an attempt to train a Convolutional Neural Network using the "Lemon Leaf Disease Dataset"
# As the dataet had only around 1300 images, it was not sufficient to train a model from scratch
# This approach has been deprecated for the project

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 256
BATCH_SIZE = 32
TRAIN_DATA_DIR = 'datasets/Prepared/lemon-leaf-disease-dataset/train/'
TEST_DATA_DIR = 'datasets/Prepared/lemon-leaf-disease-dataset/test/'
VALIDATE_DATA_DIR = 'datasets/Prepared/lemon-leaf-disease-dataset/validation/'

# Loading Training Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Loading Validation Dataset
validate_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATE_DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Loading Testing Dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Performing Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Building our Convolution NN Model
model = models.Sequential([
    data_augmentation, # data augmentation
    layers.Rescaling(1./255), # normalization

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, validation_data=validate_ds, epochs=20)

# Testing the model
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2f}")

# Saving our model
model.save("models/main-06.keras")