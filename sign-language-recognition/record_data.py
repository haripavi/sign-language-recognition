import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define labels for gestures
gestures = ["Hello", "Yes", "No", "Thank You", "Please", "Super", "Namasthe", "Salute", "Call me"]
current_label = 0

# Open webcam
cap = cv2.VideoCapture(0)
data = []

print("Press 'n' to record next gesture, 'q' to quit.")

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
            landmark_data = []
            for lm in hand_landmarks.landmark:
                landmark_data.extend([lm.x, lm.y, lm.z])
            landmark_data.append(current_label)
            data.append(landmark_data)
    
    # Display label
    cv2.putText(frame, f"Recording: {gestures[current_label]}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        current_label += 1
        if current_label >= len(gestures):
            break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save data to CSV
df = pd.DataFrame(data)
df.to_csv("data/sign_language_data.csv", index=False, header=False)
print("✅ Data collection complete. Saved to 'data/sign_language_data.csv'")
