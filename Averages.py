import cv2

import numpy as np

import UltraLibrary as ul

cap = cv2.VideoCapture('Videos\\3 - A.mp4')
    


ret = True

cv2.namedWindow( "Bad", cv2.WINDOW_AUTOSIZE );
cv2.namedWindow( "Good",cv2.WINDOW_AUTOSIZE );

runSum = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        badImg = ul.stripFrame(gframe)

        """
        Transform/edit the frame
        """
        runSum += np.sum(badImg)/640/480

        cv2.imshow("Bad", frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break


cv2.waitKey(0)
cv2.destroyAllWindows()

average = runSum/450
