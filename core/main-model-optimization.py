import tensorflow as tf

# Loading the model
model = tf.keras.models.load_model("models/lemon-leaf-disease-detector.keras")

# Loading the test dataset for Representative Dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    "datasets/Prepared/lemon-leaf-disease-dataset/test",
    image_size=(224, 224),
    batch_size=1,
)

normalization_layer = tf.keras.layers.Rescaling(1./255)

# Representative Dataset
def representative_dataset():
    for batch in dataset.take(100):  # 100 samples 
        images, _ = batch
        normalized = normalization_layer(images)
        yield [tf.cast(normalized, tf.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set the representative dataset
converter.representative_dataset = representative_dataset

# Set the supported operations to int8 only
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Ensure input/output are also int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_model = converter.convert()

# Save the model
with open("../web/static/models/lemon-leaf-disease-detector.tflite", "wb") as f:
    f.write(tflite_model)