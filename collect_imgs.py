import os
import cv2

# Base directory for storing data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Class labels for the dataset
classes = ['hello', 'thanks', 'yes', 'no', 'iloveyou', 'B', 'L']
dataset_size = 100  # Number of images per class

# Try different camera indices until a working one is found
for camera_index in range(3):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        break
else:
    print("Error: Unable to open camera.")
    exit()

# Loop through each class and collect images
for class_name in classes:
    # Create a directory for the current class
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class: {class_name}')
    print('Press "Q" to start capturing images.')

    # Wait for the user to press 'Q' to start
    while cv2.waitKey(25) != ord('q'):
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            exit()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

    # Capture images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the captured frame in the class directory
        file_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(file_path, frame)
        counter += 1

    print(f'Data collection for class "{class_name}" completed.')

cap.release()
cv2.destroyAllWindows()
