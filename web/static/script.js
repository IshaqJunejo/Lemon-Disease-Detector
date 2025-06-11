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

const input = document.querySelector('#imageInput');
const preview = document.querySelector("#preview");
const sendButton = document.querySelector("#send-to-api");

// Show preview when an image is selected
input.addEventListener('change', () => {
    console.log("File Input Changed");
    const file = input.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = e => {
            preview.src = e.target.result;
            preview.style.display = 'block';
            sendButton.style.display = 'block';
        };
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

async function uploadImage() {
    const file = input.files[0];

    if (!file) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('responseArea').textContent = processResponse(data);
    })
    .catch(error => {
        document.getElementById('responseArea').textContent = "Error: " + error;
    });

}