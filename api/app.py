from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
from PIL import Image
import io

binary_model = None
main_model = None

def ensure_model_loaded():
    global binary_model, main_model
    # Making sure binary_model is loaded
    if binary_model is None:
        print("Downloading Binary Model ... ")
        binary_model_path = hf_hub_download(repo_id="ishaquejunejo/lemon-disease-detector", filename="models/lemon-leaf-or-not.keras")
        print(f"Binary model downloaded to: {binary_model_path}")
        print("Loading Binary Model ... ")
        binary_model = tf.keras.models.load_model(binary_model_path)
    
    # Making sure main_model is loaded
    if main_model is None:
        print("Downloading Main Model ... ")
        main_model_path = hf_hub_download(repo_id="ishaquejunejo/lemon-disease-detector", filename="models/lemon-leaf-disease-detector.keras")
        print(f"Main Model downloaded to: {main_model_path}")
        print("Loading Main Model ... ")
        main_model = tf.keras.models.load_model(main_model_path)

IMG_SIZE = (224, 224)

# Processing the Image
def preprocess(image):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Method for prediction of the image
def predict_from_image_file(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "Invalid image input"}

    input_data = preprocess(image)

    leaf_prob = binary_model.predict(input_data)
    if leaf_prob[0][0] < 0.5:
        return jsonify({"Prediction": "Doesn't Look like a Lemon Leaf"})

    disease_prob = main_model.predict(input_data)

    return jsonify({"Prediction": disease_prob.tolist()})

# Running the App
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    ensure_model_loaded()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_bytes = file.read()
    prediction = predict_from_image_file(image_bytes)
    return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
