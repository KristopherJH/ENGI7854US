import cv2
import numpy as np
import UltraLibrary as UL

def avgVidIntensity(video):

    cap = cv2.VideoCapture(video)

    ret = True
    i = 0
    frameMean = np.zeros(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while ret:
        ret, frame = cap.read()

        if ret:
            gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sframe = UL.stripFrame(gframe)
            frameMean[i] = np.mean(sframe)
            i += 1

    cap.release()
    cv2.destroyAllWindows()

    minMean = np.amin(frameMean)
    maxMean = np.amax(frameMean)

    return (frameMean, minMean, maxMean)


def avgImIntensity(image):
    gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    us = UL.stripFrame(gimage)
    frameMean = np.mean(us)
    return frameMean

V1A = 'Videos/1-A.mp4'
f1AM, f1AMin, f1AMax = avgVidIntensity(V1A)
print(V1A)
print(f1AMin)
print(f1AMax)


