from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)
CORS(app) # Enabling Cross-Origin Requests

IMG_SIZE = (224, 224)

# File Size Limit: 5 MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


# Load TFLite models
interpreter1 = tf.lite.Interpreter(model_path="models/lemon-leaf-or-not.tflite")
interpreter2 = tf.lite.Interpreter(model_path="models/lemon-leaf-disease-detector.tflite")

interpreter1.allocate_tensors()
interpreter2.allocate_tensors()

# Get input/output details
input_details_1 = interpreter1.get_input_details()
output_details_1 = interpreter1.get_output_details()

input_details_2 = interpreter2.get_input_details()
output_details_2 = interpreter2.get_output_details()

def preprocess_image(file):
    img = image.load_img(io.BytesIO(file.read()), target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

def run_tflite_inference(interpreter, input_details, output_details, img_array):
    input_data = img_array

    # Quantize input if needed
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        input_data = (input_data / scale + zero_point).astype(np.int8)
    elif input_details[0]['dtype'] == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        input_data = (input_data / scale + zero_point).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output if needed
    if output_details[0]['dtype'] in [np.int8, np.uint8]:
        scale, zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - zero_point) * scale

    return output_data


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    # Validating the file is an image
    if not file or not file.mimetype.startswith('image/'):
        return jsonify({"error": "Invalid image upload"}), 400

    try:
        img_array = preprocess_image(file)

        # predicting using the binary model
        prob_of_leaf = run_tflite_inference(interpreter1, input_details_1, output_details_1, img_array)

        if prob_of_leaf[0][0] <= 0.5:
            return jsonify({"Prediction":"Doesn't Look like a Lemon Leaf"})
        else:
            # predicting using the main model
            prob_of_disease = run_tflite_inference(interpreter2, input_details_2, output_details_2, img_array)
            return jsonify({"Prediction": prob_of_disease.tolist()})

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)