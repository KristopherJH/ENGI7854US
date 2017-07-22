import cv2

import numpy as np


cap = cv2.VideoCapture('Videos\\3 - A.mp4')
    


ret = True

cv2.namedWindow( "Bad", cv2.WINDOW_AUTOSIZE );
cv2.namedWindow( "Good",cv2.WINDOW_AUTOSIZE );

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        badImg = stripFrame(gframe)

        """
        Transform/edit the frame
        """


        cv2.imshow("Bad", frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

