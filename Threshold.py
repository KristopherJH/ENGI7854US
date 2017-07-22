import cv2
import numpy as np

def avgVidIntensity(video):

    cap = cv2.VideoCapture(video)

    ret = True
    i = 0
    frameMean = np.zeros(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while ret:
        ret, frame = cap.read()

        if ret:
            gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameMean[i] = np.mean(gframe)
            cv2.imshow('Test', gframe)
            i += 1

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frameMean

fM = avgVidIntensity('Videos/5 - A.mp4')
print(fM)
