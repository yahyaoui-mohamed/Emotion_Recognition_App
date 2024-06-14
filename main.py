import threading
import tkinter as tk
import cv2
import numpy as np
import math
from keras.models import load_model
from playsound import playsound
from tkinter import ttk
from PIL import Image, ImageTk


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
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
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
                emotion_label = self.class_names[np.argmax(predictions)]

                img_path = "./avatars/"+emotion_label+".png"
                new_image = Image.open(img_path)
                new_image = new_image.resize((150, 150))
                new_photo = ImageTk.PhotoImage(new_image)
                img_label.configure(image=new_photo)
                img_label.image = new_photo
                # root.after(100, self.change_emotion) 
                progressbar1['value'] = math.ceil(predictions[0][0] * 100)
                progressbar2['value'] = math.ceil(predictions[0][1] * 100)
                progressbar3['value'] = math.ceil(predictions[0][2] * 100)
                progressbar4['value'] = math.ceil(predictions[0][3] * 100)
                progressbar5['value'] = math.ceil(predictions[0][4] * 100)
                progressbar6['value'] = math.ceil(predictions[0][5] * 100)
                progressbar7['value'] = math.ceil(predictions[0][6] * 100)

                root.update()
                root.after(50)
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
        # playsound('hello.mp3')
        # playsound('function.mp3')
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

label1 = tk.Label(root, text="Angry",bg="black",fg="white")
label1.place(x=30, y=70)
progressbar1 = ttk.Progressbar(mode="determinate")
progressbar1.place(x=30, y=100, width=200)


label2 = tk.Label(root, text="Disgust",bg="black",fg="white")
label2.place(x=30, y=170)
progressbar2 = ttk.Progressbar(mode="determinate")
progressbar2.place(x=30, y=200, width=200)

label2 = tk.Label(root, text="Fear",bg="black",fg="white")
label2.place(x=30, y=270)
progressbar3 = ttk.Progressbar(mode="determinate")
progressbar3.place(x=30, y=300, width=200)

label2 = tk.Label(root, text="Happy",bg="black",fg="white")
label2.place(x=30, y=370)
progressbar4 = ttk.Progressbar(mode="determinate")
progressbar4.place(x=30, y=400, width=200)


label2 = tk.Label(root, text="Neutral",bg="black",fg="white")
label2.place(x=30, y=470)
progressbar5 = ttk.Progressbar(mode="determinate")
progressbar5.place(x=30, y=500, width=200)


label2 = tk.Label(root, text="Sad",bg="black",fg="white")
label2.place(x=30, y=570)
progressbar6 = ttk.Progressbar(mode="determinate")
progressbar6.place(x=30, y=600, width=200)


label2 = tk.Label(root, text="Surprise",bg="black",fg="white")
label2.place(x=30, y=670)
progressbar7 = ttk.Progressbar(mode="determinate")
progressbar7.place(x=30, y=700, width=200)

img = Image.open("./avatars/neutral.jpg")
img = img.resize((150, 150))
photo = ImageTk.PhotoImage(img)
img_label = tk.Label(root, image=photo)
img_label.image = photo 
img_label.place(x=250, y=60)

play_audio()

root.mainloop()