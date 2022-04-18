from tkinter import *

import numpy as np
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from detect import Detect

root = Tk()
root.geometry("1300x800")
root.configure(bg="black")
MainFrame = LabelFrame(root, fg='white', bg="black")
MainFrame.pack(fill="both", expand=True)

Label(MainFrame, text="Special Topics in Computer Science Project", font=("Cairo", 30, "bold"), bg="black",
      fg="white").pack(side="left")

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

detect = Detect(ageModel, ageProto, genderModel, genderProto, faceModel, faceProto)

cap = 0


def startCamera():
    global cap
    cap = cv2.VideoCapture(0)


def StopCamera():
    global cap
    if cap != 0:
        cap.release()
        cap = 0


ActionFrame = LabelFrame(MainFrame, bg="green")
ActionFrame.pack(side="right", fill="both", expand=True)

StartButton = Button(ActionFrame, text="Start", bg="green", command=startCamera)
StartButton.pack(side="top", fill="both", expand=True)

ExitButton = Button(ActionFrame, text="Stop", bg="red", command=StopCamera)
ExitButton.pack(side="top", fill="both", expand=True)

Gender = Label(ActionFrame, text="Age Output :")
Gender.pack(side="top", fill="both", expand=True)

f1 = LabelFrame(root, text="Video Cameras", fg='black', bg="white", height=1000)
f1.pack(fill="both", expand=True)

L1 = Label(f1, bg="white")
L1.pack(side="left")

L2 = Label(f1, bg="black")
L2.pack(side="right")

figure = Figure()
figure_canvas = FigureCanvasTkAgg(figure, root)
figure_canvas.get_tk_widget().pack(side="bottom")

while True:
    if cap != 0:
        if cap.isOpened():
            hasFrame, frame = cap.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            frameFace, bboxes = detect.getFaceBox(detect.faceNet, small_frame)

            frameFace, bboxes, label = detect.getGenderAge(small_frame, frameFace, bboxes, frame)

            Gender.config(text=label)

            # Get The Threshold and Draw Contour on the Image
            ret, thresh = cv2.threshold(small_frame, 150, 255, cv2.THRESH_BINARY)
            im_bw = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(image=im_bw, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

            # Draw the Contour on the Image
            image_copy = small_frame.copy()
            cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                             lineType=cv2.LINE_AA)

            # Transform cv2 Image to PIL IMAGE
            image_copy = Image.fromarray(image_copy)
            image_copy = ImageTk.PhotoImage(
                image_copy.resize((int(f1.winfo_width() / 2), 400), Image.ANTIALIAS))

            # Draw the Histogram with the Colors of the Image.
            color = ('b', 'g', 'r')
            figure.clear()
            axes = figure.add_subplot()
            for i, color in enumerate(color):
                his = cv2.calcHist([frameFace], [i], None, [256], [0, 256])
                axes.plot(his, color=color)
            # Update the Histogram
            figure_canvas.draw_idle()

            # Convert the image with the detection to RGB
            frameFace = cv2.cvtColor(frameFace, cv2.COLOR_BGR2RGB)

            # Convert the image with the detection to PIL IMAGE
            image = Image.fromarray(frameFace)
            frameFace = ImageTk.PhotoImage(image.resize((int(f1.winfo_width() / 2), 400), Image.ANTIALIAS))

            # Update both Cameras
            L1["image"] = frameFace
            L2["image"] = image_copy
    else:
        # Add Gray & Black Image on the Canvas
        array = np.arange(0, int(f1.winfo_width() / 2) * 400, 1, np.uint8)
        array.fill(200)
        array = np.reshape(array, (400, int(f1.winfo_width() / 2)))
        img = Image.fromarray(array)
        frameFace = ImageTk.PhotoImage(img)
        L1["image"] = frameFace
        array.fill(0)
        array = np.reshape(array, (400, int(f1.winfo_width() / 2)))
        img = Image.fromarray(array)
        frameFace = ImageTk.PhotoImage(img)
        L2["image"] = frameFace
    root.update()
