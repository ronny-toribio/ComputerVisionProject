#!/usr/bin/python3
"""
@authors: Ronny Toribio, Kadir Altunel, Michael Cook-Stahl
@project: Computer Vision Analysis
"""
from sys import argv
from os.path import exists
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
OUTPUT_FPS = 60
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
COLORS = [(0, 0, 255),     # anger is red (BGR)
          (0, 255, 0),     # disgust is green
          (200, 200, 200), # fear is gray
          (0, 255, 255),   # happy is yellow
          (255, 255, 255), # neutral is white
          (255, 0, 0),     # sad is blue
          (0, 165, 255)]   # surprise is orange

def main(video_path):
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
    model = load_model("cnn_model.h5")

    # main loop
    while True:
        (grabbed, raw_frame) = vc.read()
        if not grabbed:
            break
        
        # process frame
        frame_confidences = []
        frame = cv2.resize(raw_frame.copy(), (48, 48))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, axis=2)
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype("float32")
        frame = frame / 255.0
        y_pred = model.predict(frame)
        emotion = np.argmax(y_pred)

        # annotate frame
        emotion_str = EMOTIONS[emotion]
        emotion_color = COLORS[emotion]
        cv2.putText(raw_frame, emotion_str, (100, 100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, emotion_color)
        vw.write(raw_frame)
        
        # frame data
        frame_data.loc[len(frame_data.index)] = y_pred[0]

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
    if len(argv) == 2:
        if exists(argv[1]):
            main(argv[1])
