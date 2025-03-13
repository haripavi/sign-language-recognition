import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import threading

# Load model
model = tf.keras.models.load_model("model/sign_language_model.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Speech Engine
engine = pyttsx3.init()

# Gesture Labels
gestures = ["Hello", "Yes", "No", "Thank You", "Please", "Super", "Namasthe", "Salute", "Call me"]

# Start webcam
cap = cv2.VideoCapture(0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])
            
            # Prediction
            data = np.array(data).reshape(1, -1)
            prediction = model.predict(data, verbose=0)
            predicted_label = gestures[np.argmax(prediction)]

            # Display prediction
            cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Speak prediction
            threading.Thread(target=speak, args=(predicted_label,)).start()

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
