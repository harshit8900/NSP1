import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import messagebox

# Load pre-trained emotion recognition model
model = load_model('emotion_model.h5')  # you need a model file!

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def start_detection():
    cap = cv2.VideoCapture(0)  # Start webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = grayscale[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float')/255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            prediction = model.predict(roi_gray)
            max_index = int(np.argmax(prediction))
            predicted_emotion = emotion_labels[max_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow('Real-Time Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Minimal frontend using Tkinter
root = tk.Tk()
root.title("Emotion Recognition System")

start_button = tk.Button(root, text="Start Detection", command=start_detection, height=2, width=20, bg='green', fg='white')
start_button.pack(pady=20)

exit_button = tk.Button(root, text="Exit", command=root.destroy, height=2, width=20, bg='red', fg='white')
exit_button.pack(pady=10)

root.mainloop()
