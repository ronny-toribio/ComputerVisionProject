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
from mtcnn import MTCNN

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
MTCNN_CONFIDENCE = 0.9
OUTPUT_FPS = 60


def main(video_path):
    video_path_base = ".".join(video_path.split(".")[:-1])
    frame_data = pd.DataFrame({
        "detections": [],
        "avg_confidence": []
    })
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        vc.release()
        return
    vw = cv2.VideoWriter(video_path_base + "_output.mp4", FOURCC, OUTPUT_FPS, (int(vc.get(3)), int(vc.get(4))))
    face_detector = MTCNN()
    detection_color = (0, 255, 0)

    # main loop
    while True:
        (grabbed, frame) = vc.read()
        if not grabbed:
            break
        
        # process frame
        frame_confidences = []
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = face_detector.detect_faces(frame)
        detection_count = len(detections)
        for det in detections:
            frame_confidences.append(det["confidence"])
            if det["confidence"] >= MTCNN_CONFIDENCE:
                x, y, width, height = det["box"]
                keypoints = det["keypoints"]
                cv2.rectangle(frame, (x, y), (x + width, y + height), detection_color, 2)
                cv2.circle(frame, (keypoints["left_eye"]), 2, detection_color, 2)
                cv2.circle(frame, (keypoints["right_eye"]), 2, detection_color, 2)
                cv2.circle(frame, (keypoints["nose"]), 2, detection_color, 2)
                cv2.circle(frame, (keypoints["mouth_left"]), 2, detection_color, 2)
                cv2.circle(frame, (keypoints["mouth_right"]), 2, detection_color, 2)
                cv2.putText(frame, str(det["confidence"]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_color)
        vw.write(frame)

        # frame data
        if detections:
            frame_data.loc[len(frame_data.index)] = [detection_count, mean(frame_confidences)]
        else:
            frame_data.loc[len(frame_data.index)] = [0, 0.0]

    # opencv cleanup
    vc.release()
    vw.release()
    cv2.destroyAllWindows()

    # plot and save frame stats
    frame_data.to_csv(video_path_base + "_stats.csv")
    plt.plot(frame_data["detections"], "r-", label = "detections")
    plt.plot([x * 100 for x in frame_data["avg_confidence"]], "b-", label = "confidence")
    plt.legend()
    plt.title("Frame Data")
    plt.savefig(video_path_base + "_stats.png")


if __name__ == "__main__":
    if len(argv) == 2:
        if exists(argv[1]):
            main(argv[1])
