import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
from playsound import playsound
from tkinter import ttk
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

model = load_model('./emotion_recognition_model.h5')

class CameraApp:
    def __init__(self, window, video_source=0, width=640, height=480):
        self.window = window
        self.video_source = video_source
        self.width = width
        self.height = height

        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.model = load_model('./emotion_recognition_model.h5')
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.update()
        self.window.mainloop()

    def preprocess_face(self, face):
        face = cv2.resize(face, (48, 48))  # Resize to the input size of your model
        # face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)  # Convert to grayscale if required
        face = face / 255.0  # Normalize
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        return face

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                processed_face = self.preprocess_face(face)
                predictions = self.model.predict(processed_face)
                emotion = self.class_names[np.argmax(predictions)]
                score = np.max(predictions)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'{emotion}: {score:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

def play_audio():
    def play_both():
        playsound('hello.mp3')
        playsound('function.mp3')
        start_camera()

    threading.Thread(target=play_both).start()

def start_camera():
    camera_app = CameraApp(root)

root = tk.Tk()
root.title("Tkinter Test")

root.configure(bg="black")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

root.geometry(f"{screen_width}x{screen_height}")

progressbar1 = ttk.Progressbar(mode="determinate")
progressbar1.place(x=30, y=60, width=200)

progressbar2 = ttk.Progressbar(mode="determinate")
progressbar2.place(x=30, y=90, width=200)

progressbar3 = ttk.Progressbar(mode="determinate")
progressbar3.place(x=30, y=120, width=200)

progressbar4 = ttk.Progressbar(mode="determinate")
progressbar4.place(x=30, y=150, width=200)

play_audio()

root.mainloop()