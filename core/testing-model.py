import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = load_model('models/lemon-leaf-disease-detector.keras')

test_generator = ImageDataGenerator(
    rescale=1./255
)

test_data = test_generator.flow_from_directory(
    'datasets/Prepared/lemon-leaf-disease-dataset/test',
    target_size=(224,224),
    batch_size=1,
    class_mode='sparse',
    shuffle=False
)

pred_prob = model.predict(test_data)
pred = np.argmax(pred_prob, axis=1)
true_y = test_data.classes
class_names = list(test_data.class_indices.keys())

print("Classification Report:")
print(classification_report(true_y, pred))

print()
print("Confusion Matrix:")
print(confusion_matrix(true_y, pred))

# Plotting the confusion matrix
plt.figure(figsize=(18, 18))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(true_y, pred), display_labels=class_names)
disp.plot(cmap=plt.cm.Reds)
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.show()