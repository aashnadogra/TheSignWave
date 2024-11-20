# train_classifier.py

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# Define and train the model (MLP classifier)
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
y_pred_train = model.predict(x_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print('Training Accuracy:', train_accuracy)

y_pred_test = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('Testing Accuracy:', test_accuracy)

# Save the trained model
with open('model_nn.p', 'wb') as file:
    pickle.dump({'model': model}, file)
