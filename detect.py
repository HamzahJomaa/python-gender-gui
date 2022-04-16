import cv2
import numpy as np


class Detect:
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    MODEL_MEAN_VALUES = 0
    ageNet = 0
    genderNet = 0
    faceNet = 0
    padding = 20
    hasFrame = False
    camera = 0
    label = ""

    def __init__(self, ageModel, ageProto, genderModel, genderProto, faceModel, faceProto):
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)
        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)

    def getFaceBox(self, net, frame, conf_threshold=0.75):
        frame = np.asarray(frame)
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

        return frameOpencvDnn, bboxes



    def getGenderAge(self, small_frame, frameFace, bboxes,frame):
        if not bboxes:
            print("No face Detected, Checking next frame")
        for bbox in bboxes:
            face = small_frame[max(0, bbox[1] - self.padding):min(bbox[3] + self.padding, frame.shape[0] - 1),
                   max(0, bbox[0] - self.padding):min(bbox[2] + self.padding, frame.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            self.genderNet.setInput(blob)
            genderPreds = self.genderNet.forward()
            gender = self.genderList[genderPreds[0].argmax()]
            # print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            age = self.ageList[agePreds[0].argmax()]

            # print("Age Output : {}".format(agePreds))
            # print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            self.label = "{},{}".format(gender, age)

            cv2.putText(frameFace, self.label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_AA)
        return frameFace, bboxes, self.label
