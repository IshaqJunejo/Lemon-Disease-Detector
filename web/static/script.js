const LABELS = [
    'Anthracnose',
    'Bacterial Blight',
    'Citrus Canker',
    'Curl Virus',
    'Deficiency Leaf',
    'Dry Leaf',
    'Healthy Leaf',
    'Sooty Mould',
    'Spider Mites'
];

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 Megabytes

const input = document.querySelector('#imageInput');
const preview = document.querySelector("#preview");
const sendButton = document.querySelector("#analyze-image");
const responseArea = document.querySelector('#responseArea');

// Loading the Models
let binaryModel;
let mainModel;

// tf.env().set('WASM_PATH', 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.14.0/dist/');

// tf.setBackend('wasm').then(() => {
//     console.log('Backend set to wasm');

//     window.onload = async function () {
//         binaryModel = await tflite.loadTFLiteModel('models/lemon-leaf-or-not.tflite');
//         mainModel = await tflite.loadTFLiteModel('models/lemon-leaf-disease-detector.tflite');
        
//         console.log('Models Loaded');
//     };
//   });

async function loadModels() {
    binaryModel = await tf.loadLayersModel('https://raw.githubusercontent.com/IshaqJunejo/Lemon-Disease-Detector/refs/heads/web/web/models/lemon-leaf-or-not/model.json');
    mainModel = await tf.loadLayersModel('https://raw.githubusercontent.com/IshaqJunejo/Lemon-Disease-Detector/refs/heads/web/web/models/lemon-leaf-disease-detector/model.json');

    console.log("Model Loaded");
}


// Show preview when an image is selected
input.addEventListener('change', () => {
    console.log("File Input Changed");
    const file = input.files[0];

    // --- VALIDATION ---
    if (!file) {
        // no file → clear UI
        preview.style.display = 'none';
        sendButton.style.display = 'none';
        responseArea.textContent = '';
        return;
    }

    // File Type Check: must be image/*
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file (jpg, png, etc.)');
        input.value = '';
        return;
    }

    // 2) File Size Check: max 5 MB
    if (file.size > MAX_FILE_SIZE) {
        alert('File too large! Please select an image under 5 MB.');
        input.value = '';
        return;
    }

    if (file) {
        const reader = new FileReader();
        reader.onload = e => {
            preview.src = e.target.result;
            preview.style.display = 'block';
            sendButton.style.display = 'block';
        };
        responseArea.textContent = '';
        reader.readAsDataURL(file);
    } else {
        preview.src = '';
        preview.style.display = 'none';
        sendButton.style.display = 'none';
    }
});

function processResponse(pred) {
    if (typeof pred.Prediction === 'string') {
        return pred.Prediction;
    } else if (Array.isArray(pred.Prediction)) {
        let output = '';
        for (let i = 0; i < pred.Prediction[0].length; i++) {
            output += `${LABELS[i]} : ${(pred.Prediction[0][i] * 100).toFixed(2)}%\n`;
        }

        return output;
    }
}

async function analyzeImage() {
    // const file = input.files[0];

    // responseArea.textContent = "Analyzing ... ";

    // if (!file) {
    //     alert("Please select an image.");
    //     return;
    // }
    // if (!file.type.startsWith('image/')) {
    //     alert("Invalid file — please select an image");
    //     return;
    // }
    // if (file.size > MAX_FILE_SIZE) {
    //     alert("Don't upload images larger than 5 MB");
    //     return;
    // }

    // const formData = new FormData();
    // formData.append('image', file);

    // fetch('https://lemon-disease-detector-production.up.railway.app/predict', {
    //     method: 'POST',
    //     body: formData
    // })
    // .then(response => response.json())
    // .then(data => {
    //     responseArea.textContent = processResponse(data);
    // })
    // .catch(error => {
    //     responseArea.textContent = "Error: " + error;
    // });

    const file = input.files[0];
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    responseArea.textContent = "Analyzing ... ";

    const img = new Image();
    img.src = URL.createObjectURL(file);

    // Resize and draw image to 224x224 canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Quantization Values
    // const inputScale = 0.003921568859368563;
    // const inputZeroPoint = -128;

    // Preparing the Input Array
    const inputTensor = tf.browser.fromPixels(canvas)
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

    // Manually Quantizing the Input to match the Models' Quantization
    // const inputTensor = inputArray
    //     .div(tf.scalar(inputScale))       // Divide by scale
    //     .add(tf.scalar(inputZeroPoint))   // Add zero point
    //     .clipByValue(-128, 127)           // Clamp to int8 range
    //     .round()
    //     .cast('int32')                    // Casting to int type
    //     .expandDims();                    // Add batch dimension

    const outputTensor1 = await binaryModel.predict(inputTensor);

    const prob_of_leaf = await outputTensor1.data();

    console.log(prob_of_leaf);

}

loadModels();