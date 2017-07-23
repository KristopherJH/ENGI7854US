import cv2
import numpy as np
import UltraLibrary as UL
import matplotlib.pyplot as plt

img = cv2.imread('Images/4-A_frame_20.png')
gimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imageSeg = UL.stripFrame(gimage)
ft = np.fft.fft2(imageSeg)
ftshift = np.fft.fftshift(ft)
print(ftshift.shape)
mag_spec = 20*np.log(np.abs(ftshift))

plt.Figure
plt.subplot(121),plt.imshow(imageSeg, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(mag_spec, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

gaux = cv2.getGaussianKernel(485, 3)
gauy = cv2.getGaussianKernel(378, 3)
gaussian = gauy*gaux.T
print(gaussian.shape)
fft_gau = np.fft.fft2(gaussian)
fft_gau_shift = np.fft.fftshift(fft_gau)
gau_spec = np.log(np.abs(fft_gau_shift)+1)
#  filter_US = np.multiply(fft_gau_shift, ftshift)
filter_US = fft_gau_shift*ftshift
improved_mag_spec = 20*np.log(np.abs(filter_US))

f_ishift = np.fft.ifftshift(improved_mag_spec)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.Figure
plt.subplot(131),plt.imshow(img_back, cmap = 'gray')
plt.title('Regenerated Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(gau_spec, cmap = 'gray')
plt.title('Gaussian Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(improved_mag_spec, cmap = 'gray')
plt.title('Altered Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()