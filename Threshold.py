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
            i += 1

    cap.release()
    cv2.destroyAllWindows()

    minMean = np.amin(frameMean)
    maxMean = np.amax(frameMean)

    return (frameMean, minMean, maxMean)

V1A = 'Videos/1-A.mp4'
V1B = 'Videos/1-B.mp4'
V2A = 'Videos/2 - A.mp4'
V2B = 'Videos/2 - B.mp4'
V3A = 'Videos/3 - A.mp4'
V3B = 'Videos/3 - B.mp4'
V4A = 'Videos/4 - A.mp4'
V4B = 'Videos/4 - B.mp4'
V5A = 'Videos/5 - A.mp4'
V5B = 'Videos/5 - B.mp4'


f1AM, f1AMin, f1AMax = avgVidIntensity(V1A)
print(V1A)
print(f1AMin)
print(f1AMax)

f1AM, f1AMin, f1AMax = avgVidIntensity(V1B)
print(V1B)
print(f1AMin)
print(f1AMax)

f1AM, f1AMin, f1AMax = avgVidIntensity(V2A)
print(V2A)
print(f1AMin)
print(f1AMax)

f1AM, f1AMin, f1AMax = avgVidIntensity(V2B)
print(V2B)
print(f1AMin)
print(f1AMax)

f1AM, f1AMin, f1AMax = avgVidIntensity(V3A)
print(V3A)
print(f1AMin)
print(f1AMax)

f1AM, f1AMin, f1AMax = avgVidIntensity(V3B)
print(V3B)
print(f1AMin)
print(f1AMax)

f1AM, f1AMin, f1AMax = avgVidIntensity(V4A)
print(V4A)
print(f1AMin)
print(f1AMax)

f1AM, f1AMin, f1AMax = avgVidIntensity(V4B)
print(V4B)
print(f1AMin)
print(f1AMax)

f1AM, f1AMin, f1AMax = avgVidIntensity(V5A)
print(V5A)
print(f1AMin)
print(f1AMax)

f1AM, f1AMin, f1AMax = avgVidIntensity(V5B)
print(V5B)
print(f1AMin)
print(f1AMax)

