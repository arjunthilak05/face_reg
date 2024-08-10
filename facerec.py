import cv2
import os
import numpy as np
import time

# Constants
FACE_DETECTOR_PATH = 'C:/Users/Hp/Downloads/lbpcascade_frontalface_improved.xml' # Path to Haar Cascade file
DATA_DIR = 'data_with_masks'
CONFIDENCE_THRESHOLD = 0.5
SAMPLE_INTERVAL = 0.1  # Time between samples in seconds

def create_user_directory(user_name):
    user_folder = os.path.join(DATA_DIR, user_name)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def get_next_image_index(user_folder, user_name):
    existing_images = [f for f in os.listdir(user_folder) if f.startswith(f"{user_name}.") and f.endswith(".jpg")]
    if not existing_images:
        return 1
    existing_indices = [int(img.split('.')[1]) for img in existing_images]
    return max(existing_indices) + 1

def capture_and_save_faces(user_name, num_photos=50):
    user_folder = create_user_directory(user_name)
    next_image_index = get_next_image_index(user_folder, user_name)

    # Load Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return 0

    count = 0
    saved_faces = 0

    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = frame[y:y+h, x:x+w]
            face_filename = os.path.join(user_folder, f"{user_name}.{next_image_index + saved_faces}.jpg")
            cv2.imwrite(face_filename, face)
            saved_faces += 1
            print(f"Saved face image: {face_filename}")

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        count += 1
        time.sleep(SAMPLE_INTERVAL)

    cap.release()
    cv2.destroyAllWindows()
    return saved_faces

def main():
    user_name = input("Enter the person's name: ")
    num_photos = 50  # Number of photos to capture

    print(f"Preparing to capture {num_photos} photos. Press 'q' to quit.")
    saved_faces = capture_and_save_faces(user_name, num_photos)
    print(f"Saved {saved_faces} face images for {user_name}")

if __name__ == "__main__":
    main()
