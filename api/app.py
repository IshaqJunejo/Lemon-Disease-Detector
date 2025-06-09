from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)
CORS(app) # Enabling Cross-Origin Requests

model = load_model("../core/models/lemon-leaf-or-not.keras")

IMG_SIZE = (224, 224)

@app.route('/predict', methods=['POST'])
def test():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']

    img = image.load_img(io.BytesIO(file.read()), target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)

    return jsonify({"Prediction": predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)