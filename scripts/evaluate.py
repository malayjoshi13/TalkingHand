import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import operator
import cv2
from keras.models import load_model
from PIL import Image

def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    return precision, recall, f1, accuracy

# Replace these paths with your actual paths
model = load_model('weights.hdf5')
test_dataset_path = "./final_dataset/test"

# Load your model and compile it before testing
# model = ... (load your model using the appropriate method)
# model.load_weights(model_weights_path)
# model.compile(...)

# Iterate through each image in the test dataset
true_labels = []
predicted_labels = []

for label_folder in os.listdir(test_dataset_path):
    label_path = os.path.join(test_dataset_path, label_folder)
    
    if os.path.isdir(label_path):
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)

            # Load and preprocess the image 
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_roi = cv2.resize(gray, (64, 64))
            channeled_mask = cv2.merge((resized_roi,resized_roi,resized_roi))
            reshaped_mask = channeled_mask.reshape(1, 64, 64, 3)

            # Make predictions using your model
            result = model.predict(reshaped_mask) 
            
            # Then we create a dictonary to map probability position/indexing with label
            prediction_dictionary ={'A': result[0][0], 
                        'B': result[0][1], 
                        'C': result[0][2],
                        'SPACE': result[0][4],
                        'DELETE': result[0][3],
                        'D': result[0][5]
                        }    
                       
            prediction = sorted(prediction_dictionary.items(), key=operator.itemgetter(1), reverse=True)  

            # Next "prediction[0][0]" will pick up element at first position of variable "prediction"
            top_label=prediction[0][0]  

            # Append true and predicted labels for later calculation
            true_labels.append(label_folder)
            predicted_labels.append(top_label)

# Convert the lists to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Calculate precision, recall, and F1 score
precision, recall, f1, accuracy = calculate_metrics(true_labels, predicted_labels)

# Print the results
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Test accuracy:", accuracy)