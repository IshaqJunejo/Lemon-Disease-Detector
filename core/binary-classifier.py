import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score

IMG_SIZE = 224
BATCH_SIZE = 32
TRAIN_DATA_DIR = 'datasets/Prepared/Lemon-Leaf-or-not/train/'
TEST_DATA_DIR = 'datasets/Prepared/Lemon-Leaf-or-not/test/'
VALIDATE_DATA_DIR = 'datasets/Prepared/Lemon-Leaf-or-not/validate/'

# Data Augmentation (for training data)
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

# Normalization (for validation and testing data)
val_test_generator = ImageDataGenerator(
    rescale=1./255
)

# Loading training data
train_data = train_generator.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Loading testing data
test_data = val_test_generator.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Loading validation data
val_data = val_test_generator.flow_from_directory(
    VALIDATE_DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Loading MobileNetV2 Base Model
base_model = load_model('models/mobile-net-v2-base.keras')
base_model.trainable = False

# Adding Top Layers to the Base Model
avg_pool = layers.GlobalAveragePooling2D()(base_model.output)

layer1 = layers.Dense(128, activation='relu')(avg_pool)
layer2 = layers.Dropout(0.3)(layer1)
layer3 = layers.Dense(64, activation='relu')(layer2)
predict = layers.Dense(1, activation='sigmoid')(layer3)

# Connecting the Base Model and Top Layers
model = models.Model(inputs=base_model.input, outputs=predict)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(train_data, validation_data=val_data, epochs=3)

# Testing the model
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc:.2f}")

# Save the model
model.save("models/lemon-leaf-or-not.keras")