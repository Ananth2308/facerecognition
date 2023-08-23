import os
import pickle
import time
import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog

import predictors
from vr2 import name_encodings_dict

from predictors import face_rects, get_image_paths, face_encodings
import vr2

import tkinter as tk
import cv2
from PIL import Image, ImageTk
flag = 0
flag_name = 1
import FD
class WebcamApp:
    def __init__(self, window, window_title):

        self.window = window
        self.window.title(window_title)

        self.video_source = 0  # 0 for the default webcam
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=1600, height=1080,borderwidth=10, relief="solid")
        self.canvas.pack()

        self.dt = Label(window, font=('Ariel', 20),borderwidth=2, relief="solid")
        self.dt.place(anchor="n", relx=0.925, rely=0.02)
        self.my_time()

        self.image = PhotoImage(file="C:/Users/dell/Desktop/logo/logo.png")
        self.logo = Label(window, image=self.image,borderwidth=2, relief="solid")
        self.logo.pack()
        self.logo.place(anchor='n', relx=0.067, rely=0.02)

        self.head = Label(window, text='Log Entry', width=20, height=2, font=('Ariel', 25, 'bold'))
        self.head.pack()
        self.head.place(anchor="n", relx=0.5, rely=0.03)

        self.btn_stop = tk.Button(window, text='Stop', width=20, height=3, command=self.stop, bg="red", fg="white",borderwidth=3, relief="solid")
        self.btn_stop.place(anchor="center", relx=0.7, rely=0.5)

        self.btn_start = tk.Button(window, text="Start", width=20, height=3, command=self.snapshot, bg="green",
                                   fg="white",borderwidth=3, relief="solid")
        self.btn_start.place(anchor="center", relx=0.7, rely=0.4)

        self.btn_exit = tk.Button(window, text="Exit", width=20, height=3, command=self.exit_app, bg="black",
                                  fg="white",borderwidth=3, relief="solid")
        self.btn_exit.place(anchor="center", relx=0.7, rely=0.6)

        self.btn_add = tk.Button(window, text="Add", width=20, height=3, command=self.add_window, bg="orange",fg="white",borderwidth=3, relief="solid")
        self.btn_add.place(anchor="center", relx=0.7, rely=0.3)

        self.btn_new = tk.Button(window, text="Crop face", width=20, height=3, command=self.new,borderwidth=3, relief= "solid",bg = 'orange',state='disabled')
        self.btn_new.place(anchor="center", relx=0.7, rely=0.7)

        self.btn_save = tk.Button(window, text="save", width=20, height=3, command=self.save,borderwidth=3, relief= "solid",bg = 'yellow',fg = 'black')
        self.btn_save.place(anchor="center", relx=0.7, rely=0.8)


        #self.indicate = Label(window, font=('Ariel', 20),borderwidth=2, relief="solid")
        # self.indicate = self.canvas.create_oval(50,50 , 130,130 , fill="blue")
        #
        # self.canvas.tag_bind(self.indicate, "<Button-1>")

        self.is_capturing = False
        self.username_entered = False
        self.username = None
        self.saving = False
        self.image_counter = 0

        self.update()

    def my_time(self):
        self.time_string = time.strftime('%H:%M:%S %p \n %d %B')
        self.dt.config(text=self.time_string)
        self.dt.after(1000, self.my_time)


    def stop(self):
        global flag
        flag = 0



    def snapshot(self):

        global flag
        flag = 1


    def exit_app(self):
        self.vid.release()
        self.window.quit()

    def update(self):
        global flag,flag_name
        ret, frame = self.vid.read()
        if flag == 1:
            encodings = face_encodings(frame)
            names = []

            # loop over the encodings
            for encoding in encodings:

                counts = {}

                for (name, encodings) in name_encodings_dict.items():
                    counts[name] = predictors.nb_of_matches(encodings, encoding)
                # checking for the name that has max no. of counts
                if all(count == 0 for count in counts.values()):
                    name = "Unknown"

                else:
                    name = max(counts, key=counts.get)

                names.append(name)

            # constructing a rectangle over the face in the webcam and the name which had the maximum number of matches
            for rect, name in zip(face_rects(frame), names):
                x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame, (700, 550))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_frame))
            x_position = 5
            y_position = 130
            self.canvas.create_image(x_position, y_position, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)
    def detect_bounding_box(self,vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        video_capture = vid
        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
        # converting the image into grayscale to produce the digital image

        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        print('no.of faces',len(faces))
        for (x, y, w, h) in faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        return faces
    def put_face(self,video_frame):
        faces = self.detect_bounding_box(video_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face = video_frame[y:y + h, x:x + w]
        return video_frame,len(faces),faces


    def new(self):

        if not self.username_entered:
            user_name = simpledialog.askstring("Input", "Enter your name:")
            if user_name:
                self.username = user_name
                self.username_entered = True
                parent_dir = "C:/Users/dell/Desktop/cam pics"
                self.path = os.path.join(parent_dir, self.username)
                if not os.path.exists(self.path):
                    os.mkdir(self.path)
                self.image_counter = 0  # Initialize the image counter for this user

        ret, frame = self.vid.read()
        if self.username and ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                for i, (x, y, w, h) in enumerate(faces):
                    face = frame[y:y + h, x:x + w]
                    image_name = f'{self.username}_{self.image_counter}.jpg'  # Unique filename with user-specific counter
                    cv2.imwrite(os.path.join(self.path, image_name), face)
                    self.image_counter += 1  # Increment the image counter for this user

    def add_window(self):
        self.btn_new.config(state = 'normal')


    def save(self):
        self.saving = True
        tkinter.messagebox.showinfo("saving",'WAIT FOR A MINUTE')



        root_dir = "C:/Users/dell/Desktop/cam pics"
        class_names = os.listdir(root_dir)
        # grabbing the name and the root directory
        image_paths = get_image_paths(root_dir, class_names)

        name_encodings_dict = {}

        nb_current_image = 1
        for image_path in image_paths:



            print(f"Image processed {nb_current_image}/{len(image_paths)}")

            image = cv2.imread(image_path)

            encodings = face_encodings(image)
            # extracting the name of the person from the path
            name = image_path.split(os.path.sep)[-2]

            e = name_encodings_dict.get(name, [])

            e.extend(encodings)

            name_encodings_dict[name] = e
            nb_current_image += 1
            if nb_current_image == len(image_paths):
                self.saving = False

        with open("features.pickle_test", "wb") as f:
            pickle.dump(name_encodings_dict, f)


def main():
    root = tk.Tk()
    app = WebcamApp(root, "Webcam App")
    root.mainloop()

if __name__ == '__main__':
    main()

# window.mainloop()


