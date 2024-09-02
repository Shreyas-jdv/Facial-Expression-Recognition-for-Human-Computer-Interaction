import numpy as np
from emotion_classification import train_classifiers
from feature_extraction import extract_features
import cv2
import os


def load_dataset(dataset_path):
    X = []
    y_svm = []
    y_dt = []

    # Assuming dataset is organized with each subfolder representing an emotion
    emotions = os.listdir(dataset_path)
    for emotion in emotions:
        emotion_path = os.path.join(dataset_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            image = cv2.imread(img_path)
            features = extract_features(image)
            if features is not None:
                X.append(features)
                # Binary labels: 1 for positive emotions (e.g., happy), 0 for negative
                y_svm.append(1 if emotion in ['happy', 'excited'] else 0)
                # Multi-class labels
                y_dt.append(emotions.index(emotion))

    return np.array(X), np.array(y_svm), np.array(y_dt)


def main():
    dataset_path = "C:\Users\Lenovo\PycharmProjects\Facial Expression Recognition for Human-Computer Interaction\emotions.xlsx"
    X, y_svm, y_dt = load_dataset(dataset_path)
    print(f"Loaded {len(X)} samples.")
    train_classifiers(X, y_svm, y_dt)
    print("Training complete and models saved.")


if __name__ == "__main__":
    main()
