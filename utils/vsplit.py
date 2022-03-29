#!/usr/bin/python3
"""
@author: Ronny Toribio
@project: Video Splitter Utility
"""
import os
from os.path import exists, join
from sys import argv
import cv2


def video_splitter(video_path, frames_path):
    if not exists(frames_path):
        os.makedirs(frames_path, exist_ok=True)
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        vc.release()
        return
    frame_count = 0
    while True:
        (grabbed, frame) = vc.read()
        if not grabbed:
            break
        cv2.imwrite(os.path.join(frames_path, "{}.jpg".format(frame_count)), frame)
        frame_count += 1
    vc.release()


if __name__ == "__main__":
    if len(argv) == 3:
        if exists(argv[1]):
            video_splitter(argv[1], argv[2])
        else:
            print("The video {} doesn't exist.".format(video_path))
    else:
        print("{} [video] [output directory]".format(argv[0]))
