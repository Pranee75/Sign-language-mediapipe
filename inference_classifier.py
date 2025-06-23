import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.75  # You can tune this value

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label dictionary
labels_dict = {0: 'Hello', 1: 'Cat', 2: 'No',3:'I Love You',4:'Thankyou',5:'Fine',6:'B',7:'Sorry',8:'A'}

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = "unidentified"  # Default label

    if results.multi_hand_landmarks:
        hand_sizes = []
        hand_landmarks_list = []

        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x1, x2 = min(x_coords) * W, max(x_coords) * W
            y1, y2 = min(y_coords) * H, max(y_coords) * H

            hand_size = (x2 - x1) * (y2 - y1)
            hand_sizes.append(hand_size)
            hand_landmarks_list.append(hand_landmarks)

        largest_hand_index = np.argmax(hand_sizes)
        selected_hand = hand_landmarks_list[largest_hand_index]

        mp_drawing.draw_landmarks(
            frame, selected_hand, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for lm in selected_hand.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in selected_hand.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        # Bounding box
        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

        # Predict with confidence check
        try:
            probabilities = model.predict_proba([np.asarray(data_aux)])[0]
            max_prob = np.max(probabilities)
            predicted_index = np.argmax(probabilities)

            if max_prob >= CONFIDENCE_THRESHOLD:
                predicted_character = labels_dict[predicted_index]
            else:
                predicted_character = "unidentified"
        except AttributeError:
            # fallback if model doesn't support predict_proba
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

        # Draw output
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
