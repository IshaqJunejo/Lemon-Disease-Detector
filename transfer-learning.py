import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

IMG_SIZE = 224
BATCH_SIZE = 32
TRAIN_DATA_DIR = 'datasets/Prepared/lemon-leaf-disease-dataset/train/'
TEST_DATA_DIR = 'datasets/Prepared/lemon-leaf-disease-dataset/test/'
VALIDATE_DATA_DIR = 'datasets/Prepared/lemon-leaf-disease-dataset/validation/'

# Data Agumentation (+ Normalization)
train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Normalization
val_test_generator = ImageDataGenerator(
    rescale=1./255
)

# Loading training data (with data augmentation)
train_data = train_generator.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

# Loading testing and validation data (with normalization)
test_data = val_test_generator.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

val_data = val_test_generator.flow_from_directory(
    VALIDATE_DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)


# Loading MobileNetV2 Base Model
base_model = load_model('models/mobile-net-v2-base.keras')
base_model.trainable = False

# Adding Top Layers to the Base Model
avg_pool = layers.GlobalAveragePooling2D()(base_model.output)

layer1 = layers.Dense(128, activation='relu')(avg_pool)
layer2 = layers.Dropout(0.3)(layer1)
layer3 = layers.Dense(64, activation='relu')(layer2)
predict = layers.Dense(9, activation='softmax')(layer3)

# Connecting the Base Model and Top Layers
model = models.Model(inputs=base_model.input, outputs=predict)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Traing the model
history = model.fit(train_data, validation_data=val_data, epochs=15)

# Testing the model
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc:.2f}")

# Save the model
model.save("models/mobile-net-v2-full-02.keras")