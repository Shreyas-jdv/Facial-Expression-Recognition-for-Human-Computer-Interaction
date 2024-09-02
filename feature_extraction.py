import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)


def extract_features(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return None  # No face detected

    landmarks = results.multi_face_landmarks[0]
    features = []

    for lm in landmarks.landmark:
        x = lm.x
        y = lm.y
        z = lm.z
        features.extend([x, y, z])

    return features
