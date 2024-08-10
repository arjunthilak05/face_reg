import cv2
import numpy as np
from PIL import Image
import os

# Initialize the recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
FACE_DETECTOR_PATH = 'C:/Users/Hp/Downloads/lbpcascade_frontalface_improved.xml'
detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)


def imgs_and_labels(data_dir):
    faces = []
    ids = []
    labels = {}
    label_id = 0

    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        if os.path.isdir(person_path):
            labels[label_id] = person_dir
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                img = Image.open(image_path).convert('L')  # Convert to grayscale
                img_np = np.array(img, 'uint8')
                
                detected_faces = detector.detectMultiScale(img_np)
                for (x, y, w, h) in detected_faces:
                    face = img_np[y:y+h, x:x+w]
                    faces.append(face)
                    ids.append(label_id)
            
            label_id += 1

    return faces, ids, labels

# Provide the path to your data directory
data_path = 'data'  # Update this path to your data directory
faces, ids, labels = imgs_and_labels(data_path)

# Train the model
print("Training model...")
recognizer.train(faces, np.array(ids))

# Save the trained model and labels
recognizer.save('face_recognizer.yml')
np.save('face_labels.npy', labels)
print("Model trained and saved successfully!")
