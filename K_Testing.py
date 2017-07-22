import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread('test.png')

cv2.imshow('Hello World', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
Destroy specific window: cv2.destroyWindow('Window Name')
Save an image: cv2.imwrite('name.png', img)
"""


image = Image.open('test.png')
arr = np.asarray(image)
plt.imshow(arr, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()