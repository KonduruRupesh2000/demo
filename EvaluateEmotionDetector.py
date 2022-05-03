
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import adam_v2

emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
model = json_file.read()
json_file.close()
emotion_model = model_from_json(model)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
        'data/test',
        target_size=(30, 30),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# test_generator = test_generator.class_indices
# test_generator = {v: k for k, v in test_generator.items()}
# classes = list(test_generator.values())

# #Confution Matrix and Classification Report
# Y_pred = emotion_model.predict_generator(test_generator, 7067 // 64)
# y_pred = np.argmax(Y_pred, axis=1)

# print('Confusion Matrix')
# print(confusion_matrix(test_generator.classes, y_pred))
# print('Classification Report')
# target_names = list(test_generator.values())
# print(classification_report(test_generator.classes, y_pred, target_names=target_names))

# plt.figure(figsize=(8,8))
# cnf_matrix = confusion_matrix(test_generator.classes, y_pred)

# do prediction on test data
predictions = emotion_model.predict_generator(test_generator)

# # see predictions
for result in predictions:
     max_index = int(np.argmax(result))
     print(emotion_dict[max_index])

print("-----------------------------------------------------------------")
# # confusion matrix
c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_dict)
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

# # Classification report
print("-----------------------------------------------------------------")
print(classification_report(test_generator.classes, predictions.argmax(axis=1)))
confusion_matrix()




