import cv2

import numpy as np

import UltraLibrary as ul

winSize = 7 #nxn window for filtering
halfWin = winSize/2

#typical h for homogeneous region is 1.6 at window size 7x7
# take anything between 1.7 and 1.5 as homogeneous
highThresh = 1.7

sigmaX = 1
sigmaY = 1

pad = halfWin + 1 #how many pixels to pad the image

img = ul.stripFrame(cv2.imread('GoodImages\\5-A.png',0))

hMat = np.zeros(img.shape)

img = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_REFLECT)
size = img.shape
newimg = np.zeros(size)
hMat = np.zeros(size)


#generating gaussian kernel
kernelX = cv2.getGaussianKernel(winSize, sigmaX)
kernelY = cv2.getGaussianKernel(winSize, sigmaY) 
Gaussian = np.matmul(kernelX, np.transpose(kernelY))


#loop through all original pixels
for i in range(pad+1,size[0]-pad):
    for j in range(pad+1,size[1]-pad):
        W = img[i-halfWin:i+halfWin+1,j-halfWin:j+halfWin+1]
        mean = np.mean(W)
        vari = np.var(W)

        h = vari/mean

        if h> highThresh: 
            newimg[i,j] = np.median(W)
            hMat[i,j] = 1
        else:
            newimg[i,j] = np.mean(W)

        #print(i,j, newimg[i,j])

cv2.imshow('despeckled?', newimg.astype(int))
#cv2.imshow('homogeny', hMat)
cv2.waitKey(0)




"""
h0 =0 
w0 = 0
goodImg = ul.stripFrame(cv2.imread('GoodImages\\3-A.png',0))
img = ul.stripFrame(cv2.imread('GoodImages\\5-A.png',0))
im2 = ul.stripFrame(cv2.imread('GoodImages\\5-A.png',0))
size = img.shape

img = cv2.GaussianBlur(img, (3,3),sigmaX = 1);
cv2.imshow('median', img)

cv2.waitKey(0)

img = ul.global_histogram(img,goodImg)

#img = cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_REFLECT)


size = img.shape

cv2.imshow('hist_median', img)

cv2.waitKey(0)

im2 = ul.global_histogram(im2,goodImg)


cv2.imshow('no_median', im2)

cv2.waitKey(0)

"""




"""
for i in range(51,size[0]-50):
    for j in range(51,size[1]-50):
        homog = False
        rSize = 11
        while not homog:
            W = img[i-rSize/2:i+rSize/2+1,j-rSize/2:j+rSize/2+1]
            mean = np.sum(W)
            var = np.var(W)

            hij =var*var/mean

            if hij < h0:
                homog = True

"""




       
