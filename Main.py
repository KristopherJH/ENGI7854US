import cv2
import numpy as np
import UltraLibrary as UL
import FrameType as FT
from datetime import datetime

# Vector of video files for project:
vids = ['Videos/1-A.mp4', 'Videos/1-B.mp4', 'Videos/2-A.mp4', 'Videos/2-B.mp4',
        'Videos/3-A.mp4', 'Videos/3-B.mp4', 'Videos/4-A.mp4', 'Videos/4-B.mp4',
        'Videos/5-A.mp4', 'Videos/5-B.mp4', 'Videos/Varying.mp4']


def frame_delay(start):
    now = 0
    timer = datetime.now()
    now = (timer.day * 24 * 60 * 60 + timer.second) * 1000 + timer.microsecond/1000
    print(start)
    print(now)
    diff = now - start
    print(diff)
    next_frame = 33 - diff
    print(next_frame)
    if next_frame <= 1:
        return 1
    elif 1 < next_frame < 33:
        return next_frame
    else:
        return 33

good_im = cv2.imread('Images/3-A_frame_39.png')
good_i = UL.stripFrame(good_im)

empty = np.zeros((480, 640))
cv2.imshow('After', np.copy(empty))
cv2.imshow('Before', np.copy(empty))

cap = cv2.VideoCapture(vids[10])  # Open specified video file

ret = True  # Initialize ret to be True, ret keeps track if there is 1+ frames left in vid

while ret:
    start_t = 0
    delay = 0
    time = 0
    ret, frame = cap.read()  # Read in next available frame from video, ret=True if there is 1
    time = datetime.now()
    start_t = (time.day * 24 * 60 * 60 + time.second) *1000 + time.microsecond/1000

    if ret:
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale
        sframe = UL.stripFrame(gframe)  # Call stripFrame to remove outer info from US image

        type = FT.f_type(sframe)
        print(type)

        if type == '1':
            fImproved = UL.global_histogram(sframe, good_i)
            delay = frame_delay(start_t)
            print(delay)
        elif type == '2':
            fImproved = UL.global_histogram(sframe, good_i)
            delay = frame_delay(start_t)
            print(delay)
        elif type == '3':
            fImproved = UL.global_histogram(sframe, good_i)
            fImproved = sframe
            delay = frame_delay(start_t)
            print(delay)
        elif type == '4':
            fImproved = UL.global_histogram(sframe, good_i)
            delay = frame_delay(start_t)
            print(delay)
        elif type == '5':
            fImproved = UL.global_histogram(sframe, good_i)
            delay = frame_delay(start_t)
            print(delay)

        regenFrame = np.copy(gframe)
        regenFrame[33:411, 98:583] = fImproved

        cv2.imshow('After', regenFrame)
        cv2.imshow('Before', gframe)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
