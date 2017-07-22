import cv2
import numpy as np

# import matplotlib.pyplot as plt
# from PIL import Image

"""
Load and display image:
img = cv2.imread('test.png')
cv2.imshow('Hello World', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
Destroy specific window: 
cv2.destroyWindow('Window Name')
Save an image:
v2.imwrite('name.png', img)
"""

"""
image = Image.open('test.png')
arr = np.asarray(image)
plt.imshow(arr, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()
"""

cap = cv2.VideoCapture('Videos/1-A.mp4')

ret = True

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        desFrame = gframe[33:411, 98:583]

        """
        Transform/edit the frame
        """

        regenFrame = gframe
        regenFrame[33:411, 98:583] = desFrame

        cv2.imshow('frame', regenFrame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
