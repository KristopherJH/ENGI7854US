import cv2
import numpy as np
import UltraLibrary as UL
import FrameType as FT

# Vector of video files for project:
vids = ['Videos/1-A.mp4', 'Videos/1-B.mp4', 'Videos/2-A.mp4', 'Videos/2-B.mp4',
        'Videos/3-A.mp4', 'Videos/3-B.mp4', 'Videos/4-A.mp4', 'Videos/4-B.mp4',
        'Videos/5-A.mp4', 'Videos/5-B.mp4', 'Videos/Varying']


cap = cv2.VideoCapture(vids[1])  # Open specified video file

ret = True  # Initialize ret to be True, ret keeps track if there is 1+ frames left in vid

while ret:
    ret, frame = cap.read()  # Read in next available frame from video, ret=True if there is 1

    if ret:
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale
        sframe = UL.stripFrame(gframe)  # Call stripFrame to remove outer info from US image

        type = FT.f_type(sframe)

        regenFrame = np.copy(gframe)
        regenFrame[33:411, 98:583] = sframe

        cv2.imshow('After', regenFrame)
        cv2.imshow('Before', gframe)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()