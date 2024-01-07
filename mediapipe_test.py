import tensorflow as tf
import cv2
import mediapipe as mp
from collections import deque
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
TF_ENABLE_ONEDNN_OPTS=0
CLASSES = ["airplane", "apple", "bee", "ice cream", "tree"]

model = tf.keras.models.load_model(r"C:\Users\Admin_PC\SML-Final\my_model.h5")

cap = cv2.VideoCapture(0)
points = deque(maxlen=512)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
is_drawing = False
is_shown = False
with mp_hands.Hands(max_num_hands = 1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y and hand_landmarks.landmark[11].y and hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:
          if len(points):
            is_drawing = False
            is_shown = True
            canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            canvas_gs = cv2.medianBlur(canvas, 9)
            canvas_gs = cv2.GaussianBlur(canvas_gs, (5,5), 0)
            print(np.nonzero(canvas_gs))
            ys, xs = np.nonzero(canvas_gs)
            if len(ys) and len(xs):
              min_y = np.min(ys)
              max_y = np.max(ys)
              min_x = np.min(xs)
              max_x = np.max(xs)
              cropped_image = canvas_gs[min_y:max_y, min_x:max_x]
              cropped_image = cv2.resize(cropped_image, (28,28))
              cropped_image = np.expand_dims(cropped_image, axis = 0)
              cropped_image = tf.convert_to_tensor(cropped_image)
              int_pred = model.predict(cropped_image)
              sec_pred = np.argmax(int_pred)
              points = deque(max_len=512)
              canvas = np.zeros((480, 640, 3), dtype=np.uint8)
          else:
            is_drawing = True
            is_shown = False
            points.append((int(hand_landmarks.landmark[8].x*640), int(hand_landmarks.landmark[8].y*480)))
            for i in range(1, len(points)):
              cv2.line(image, points[i - 1], points[i], (0,255,0), 2)
              cv2.line(canvas, points[i - 1], points[i], (255,255,255), 5)
          mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
          if not is_drawing and is_shown:
            cv2.putText("You are drawing " + CLASSES[sec_pred], (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5,cv2.LINE_AA)
          




      
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
      
cap.release()