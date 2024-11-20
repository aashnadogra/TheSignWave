import os
import cv2
import pickle
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('./model_nn.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands module
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels list
labels = ['A', 'B', 'C', 'iloveyou', 'B', 'L']

# Define the camera index range to try
camera_index_range = range(3)

# Attempt to open the camera
for camera_index in camera_index_range:
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        break
else:
    # If none of the camera indices worked, print an error message and exit
    print("Error: Unable to open camera.")
    exit()

# Main loop to capture and process frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if frame reading was successful
    if not ret:
        print("Error: Unable to read frame from the camera.")
        break

    # Get the height and width of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB (MediaPipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Check if hand landmarks were detected in the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Prepare hand landmarks for prediction
            data_aux = []
            x_ = []
            y_ = []
            for landmark in hand_landmarks.landmark:
                # Normalize landmark coordinates
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            for landmark in hand_landmarks.landmark:
                # Normalize and scale landmark coordinates
                x = (landmark.x - min_x) / (max_x - min_x)
                y = (landmark.y - min_y) / (max_y - min_y)
                data_aux.extend([x, y])

            # Make a prediction using the model
            prediction = model.predict([data_aux])
            predicted_index = int(prediction[0])  # Convert prediction to integer
            predicted_letter = labels[predicted_index]

            # Draw bounding box and predicted letter on the frame
            cv2.putText(frame, predicted_letter, (int(min_x * W), int(min_y * H) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the processed frame
    cv2.imshow('frame', frame)
    
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
