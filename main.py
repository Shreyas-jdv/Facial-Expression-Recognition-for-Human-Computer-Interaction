from feature_extraction import extract_features
from emotion_classification import svm_classification, decision_tree_classification, load_classifiers
from video_processing import process_video
import cv2


def main():
    # Load pre-trained classifiers
    try:
        load_classifiers()
    except:
        print("Models not found. Please train the classifiers first.")
        return

    # Initialize video capture (webcam)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features from the frame
        features = extract_features(frame)

        if features is not None:
            # Classify emotion using SVM and Decision Tree
            emotion_svm = svm_classification(features)
            emotion_dt = decision_tree_classification(features)

            # Map numeric labels to emotion names
            emotions = ['negative', 'positive']  # Update as per your labels
            emotions_dt = ['angry', 'happy', 'sad', 'surprised']  # Example for multi-class

            svm_emotion = emotions[emotion_svm]
            dt_emotion = emotions_dt[emotion_dt]

            # Display emotions on the frame
            cv2.putText(frame, f"SVM: {svm_emotion}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Decision Tree: {dt_emotion}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow('Emotion Recognition', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
