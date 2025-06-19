import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
#from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model1 = load_model('models/lemon-leaf-or-not.keras')
model2 = load_model('models/lemon-leaf-disease-detector.keras')

test_generator = ImageDataGenerator(
    rescale=1./255
)

test_data1 = test_generator.flow_from_directory(
    'datasets/Prepared/Lemon-Leaf-or-not/test',
    target_size=(224, 224),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

test_data2 = test_generator.flow_from_directory(
    'datasets/Prepared/lemon-leaf-disease-dataset/test',
    target_size=(224,224),
    batch_size=1,
    class_mode='sparse',
    shuffle=False
)

pred_prob1 = model1.predict(test_data1)
pred1 = (pred_prob1 > 0.5).astype(int)
true_y1 = test_data1.classes

pred_prob2 = model2.predict(test_data2)
pred2 = np.argmax(pred_prob2, axis=1)
true_y2 = test_data2.classes
class_names2 = list(test_data2.class_indices.keys())

print("Classification Report:")
print(classification_report(true_y1, pred1))
print()
print("Confusion Matrix:")
print(confusion_matrix(true_y1, pred1))
print()
print("F1 Score: ", f1_score(true_y1, pred1, average='weighted'))

print()
print("Classification Report:")
print(classification_report(true_y2, pred2))
print()
print("Confusion Matrix:")
print(confusion_matrix(true_y2, pred2))
print()
print("F1 Score: ", f1_score(true_y2, pred2, average='weighted'))

# # Plotting the confusion matrix
# plt.figure(figsize=(18, 18))
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(true_y, pred), display_labels=class_names)
# disp.plot(cmap=plt.cm.Reds)
# plt.title('Confusion Matrix')
# plt.xticks(rotation=90)
# plt.show()