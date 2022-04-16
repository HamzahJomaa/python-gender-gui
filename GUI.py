from tkinter import *
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from detect import Detect

root = Tk()
root.geometry("1300x800")
root.resizable(False, False)
root.configure(bg="black")
MainFrame = LabelFrame(root, fg='white', bg="black")
MainFrame.pack(fill="both", expand="yes")

Label(MainFrame, text="Special Topics in Computer Science Project", font=("Cairo", 30, "bold"), bg="black",
      fg="white").pack(side="left")

ActionFrame = LabelFrame(MainFrame, bg="green")
ActionFrame.pack(side="right",fill="both", expand="yes")



Gender = Label(ActionFrame, text="Age Output :")
Gender.pack(side="bottom")

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


B = Button(ActionFrame, text="Start", command=startCamera)
B.pack(side="top")

f1 = LabelFrame(root, text="Video Cameras", fg='black', bg="white")
f1.pack(fill="both", expand="yes")

L1 = Label(f1, bg="white")
L1.pack(side="left")

L2 = Label(f1, bg="black")
L2.pack(side="right")

f2 = LabelFrame(root, text="Video Cameras", fg='black', bg="red", height=4)
f2.pack(fill="both", expand="yes")

figure = Figure()
figure_canvas = FigureCanvasTkAgg(figure, f2)
figure_canvas.get_tk_widget().pack(expand="yes")

while True:
    if cap != 0:
        hasFrame, frame = cap.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        frameFace, bboxes = detect.getFaceBox(detect.faceNet, small_frame)
        frameFace, bboxes, label = detect.getGenderAge(small_frame, frameFace, bboxes, frame)

        Gender.config(text=label)

        ret, thresh = cv2.threshold(small_frame, 150, 255, cv2.THRESH_BINARY)
        im_bw = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=im_bw, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # draw contours on the original image
        image_copy = small_frame.copy()
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)

        image_copy = Image.fromarray(image_copy)
        image_copy = ImageTk.PhotoImage(
            image_copy.resize((int(f1.winfo_width() / 2), 400), Image.ANTIALIAS))

        color = ('b', 'g', 'r')
        figure.clear()
        axes = figure.add_subplot()
        for i, color in enumerate(color):
            his = cv2.calcHist([frameFace], [i], None, [256], [0, 256])
            axes.plot(his, color=color)
        figure_canvas.draw_idle()

        frameFace = cv2.cvtColor(frameFace, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frameFace)
        frameFace = ImageTk.PhotoImage(image.resize((int(f1.winfo_width() / 2), 400), Image.ANTIALIAS))

        L1["image"] = frameFace
        L2["image"] = image_copy

    root.update()
