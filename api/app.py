from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)
CORS(app) # Enabling Cross-Origin Requests

model1 = load_model("../core/models/lemon-leaf-or-not.keras")
model2 = load_model("../core/models/lemon-leaf-disease-detector.keras")

IMG_SIZE = (224, 224)

# File Size Limit: 5 MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

@app.route('/predict', methods=['POST'])
def test():
    file = request.files['image']

    # Validating the file is an image
    if not file or not file.mimetype.startswith('image/'):
        return jsonify({"error": "Invalid image upload"}), 400

    # processing image
    img = image.load_img(io.BytesIO(file.read()), target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # predicting using the model
    prob_of_leaf = model1.predict(img_array)

    if prob_of_leaf[0][0] <= 0.5:
        return jsonify({"Prediction":"Doesn't Look like a Lemon Leaf"})
    else:
        prob_of_disease = model2.predict(img_array)
        return jsonify({"Prediction": prob_of_disease.tolist()})

if __name__ == '__main__':
    app.run(debug=False)