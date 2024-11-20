import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Iterate over directories in the data directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    # Check if dir_path is a directory
    if os.path.isdir(dir_path):
        # Iterate over image files in the directory
        for img_path in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_path)
            
            # Read image from file
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Unable to read image '{img_path}'. Skipping...")
                continue
            
            # Convert image to RGB format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process image to detect hand landmarks
            results = hands.process(img_rgb)
            
            # Check if hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    for landmark in hand_landmarks.landmark:
                        # Extract landmark coordinates
                        x = landmark.x
                        y = landmark.y
                        data_aux.append(x)
                        data_aux.append(y)
                    data.append(data_aux)
                    labels.append(dir_)
            else:
                print(f"No hand landmarks detected in '{img_path}'. Skipping...")

        # Reinitialize hands object to release resources
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data saved to 'data.pickle' successfully.")
