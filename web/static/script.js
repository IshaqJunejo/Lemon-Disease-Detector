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
const sendButton = document.querySelector("#send-to-api");
const responseArea = document.querySelector('#responseArea');

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

async function uploadImage() {
    const file = input.files[0];

    responseArea.textContent = "Analyzing ... ";

    if (!file) {
        alert("Please select an image.");
        return;
    }
    if (!file.type.startsWith('image/')) {
        alert("Invalid file — please select an image");
        return;
    }
    if (file.size > MAX_FILE_SIZE) {
        alert("Don't upload images larger than 5 MB");
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
        responseArea.textContent = processResponse(data);
    })
    .catch(error => {
        responseArea.textContent = "Error: " + error;
    });

}