import cv2
import numpy as np
import UltraLibrary as UL
import Despeckle as DS
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


def outputFrame(original, improved, spacer):
    temp = np.append(original, improved, axis=1)
    return np.append(temp, spacer, axis=1).astype(np.uint8)

frameSplitter = np.zeros((480, 100))

warning_img = cv2.imread('Images/Warning.png')
warning_im = cv2.cvtColor(warning_img, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale

good_img_sys = cv2.imread('Images/3-A_frame_39.png')
good_im_sys = UL.stripFrame(good_img_sys)
good_i_sys = cv2.cvtColor(good_im_sys, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale

good_img_dia = cv2.imread('Images/3-A_frame_39.png')
good_im_dia = UL.stripFrame(good_img_dia)
good_i_dia = cv2.cvtColor(good_im_dia, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale


empty = np.zeros((480, 640))
outputTemplate = outputFrame(empty, frameSplitter, empty)
cv2.imshow('Original vs. Corrected', outputTemplate)

cap = cv2.VideoCapture(vids[1])  # Open specified video file

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
            heartState = DS.hist_first(sframe)
            if heartState:
                fImproved = UL.global_histogram(sframe, good_i_sys)
            else:
                fImproved = UL.global_histogram(sframe, good_i_dia)
            regenFrame = np.copy(gframe)
            regenFrame[33:411, 98:583] = fImproved
            regenFrame[324:359, 93:598] = warning_im
            delay = frame_delay(start_t)
        elif type == '2':
            fImproved = UL.global_histogram(sframe, good_i)
            delay = frame_delay(start_t)
            regenFrame = np.copy(gframe)
            regenFrame[33:411, 98:583] = fImproved
        elif type == '3':
            fImproved = sframe
            delay = frame_delay(start_t)
            regenFrame = np.copy(gframe)
            regenFrame[33:411, 98:583] = fImproved
        elif type == '4':
            fImproved = UL.global_histogram(sframe, good_i)
            delay = frame_delay(start_t)
            regenFrame = np.copy(gframe)
            regenFrame[33:411, 98:583] = fImproved
        elif type == '5':
            fImproved = UL.global_histogram(sframe, good_i)
            delay = frame_delay(start_t)
            pregenFrame = np.copy(gframe)
            regenFrame[33:411, 98:583] = fImproved

        display = outputFrame(gframe, frameSplitter, regenFrame)
        cv2.imshow('Original vs. Corrected', display)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
