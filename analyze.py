#!/usr/bin/python3
"""
@authors: Ronny Toribio
@project: Computer Vision Analysis
"""
from sys import argv
from os.path import exists
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from mtcnn import MTCNN
from tensorflow.keras.models import model_from_json


FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
OUTPUT_FPS = 30
MTCNN_CONFIDENCE = 0.5
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
COLORS = [
    (0, 0, 255),     # anger is red (BGR)
    (0, 255, 0),     # disgust is green
    (200, 200, 200), # fear is gray
    (0, 255, 255),   # happy is yellow
    (255, 255, 255), # neutral is white
    (255, 0, 0),     # sad is blue
    (0, 165, 255)    # surprise is orange
]

def main(model_json, model_weights, video_path):
    video_path_base = ".".join(video_path.split(".")[:-1])
    frame_data = pd.DataFrame({
        "angry":[], "disgust":[], "fear":[], "happy":[],
        "neutral":[], "sad":[], "surprise":[]
    })
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        vc.release()
        return
    vw = cv2.VideoWriter(video_path_base + "_output.mp4", FOURCC, OUTPUT_FPS, (int(vc.get(3)), int(vc.get(4))))

    # load model from json and weights from hdf5
    with open(model_json, "r") as json_model_file:
        json_model_str = json_model_file.read()
    model = model_from_json(json_model_str)
    model.load_weights(model_weights)

    # face detecting model
    face_detector = MTCNN()

    # main loop
    while True:
        (grabbed, raw_frame) = vc.read()
        if not grabbed:
            break
        frame = raw_frame.copy()
        angry_count = 0
        disgust_count = 0
        fear_count = 0
        happy_count = 0
        neutral_count = 0
        sad_count = 0
        surprise_count = 0

        # detect faces
        detections = face_detector.detect_faces(frame)
        for det in detections:
            x, y, width, height = det["box"]
            if det["confidence"] >= MTCNN_CONFIDENCE and width > 0 and height > 0:
                # get face from frame
                padding = 4
                face = frame[x - padding : x + width + padding*2, y - padding : y + height + padding*2]
                if face.size == 0:
                    continue
        
                # process face with our model to determine its emotion
                face = cv2.resize(face, (48, 48))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = np.expand_dims(face, axis=2)
                face = np.expand_dims(face, axis=0)
                face = face.astype("float32")
                face = face / 255.0
                y_pred = model.predict(face)
                emotion = np.argmax(y_pred)
                emotion_str = EMOTIONS[emotion]
                emotion_color = COLORS[emotion]

                # annotate face with its emotion and the corresponding color
                cv2.rectangle(raw_frame, (x, y), (x + width, y + height), emotion_color, 2)
                cv2.putText(raw_frame, emotion_str, (x, y - 5), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, emotion_color)

                # increment emotion count
                if emotion == 0:
                    angry_count += 1
                elif emotion == 1:
                    disgust_count += 1
                elif emotion == 2:
                    fear_count += 1
                elif emotion == 3:
                    happy_count += 1
                elif emotion == 4:
                    neutral_count += 1
                elif emotion == 5:
                    sad_count += 1
                elif emotion == 6:
                    surprise_count += 1

        # save frame
        vw.write(raw_frame)
        
        # frame emotions
        frame_data.loc[len(frame_data.index)] = [angry_count, disgust_count, fear_count, happy_count,
                                                 neutral_count, sad_count, surprise_count]

    # opencv cleanup
    vc.release()
    vw.release()

    # plot and save frame stats
    frame_data.to_csv(video_path_base + "_stats.csv")
    plt.plot(frame_data[EMOTIONS[0]], "r-", label = EMOTIONS[0])
    plt.plot(frame_data[EMOTIONS[1]], "m-", label = EMOTIONS[1])
    plt.plot(frame_data[EMOTIONS[2]], "g-", label = EMOTIONS[2])
    plt.plot(frame_data[EMOTIONS[3]], "b-", label = EMOTIONS[3])
    plt.plot(frame_data[EMOTIONS[4]], "c-", label = EMOTIONS[4])
    plt.plot(frame_data[EMOTIONS[5]], "k-", label = EMOTIONS[5])
    plt.plot(frame_data[EMOTIONS[6]], "y-", label = EMOTIONS[6])
    plt.legend()
    plt.title("Frame Data")
    plt.savefig(video_path_base + "_stats.png")


if __name__ == "__main__":
    if len(argv) == 3:
        if exists(argv[1]+ ".json") and exists(argv[1] + ".h5") and exists(argv[2]):
            main(argv[1] + ".json", argv[1]+".h5", argv[2])
