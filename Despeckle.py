import cv2
import matplotlib.pyplot as plt
import numpy as np

import UltraLibrary as ul

from FrameType import f_type

def hist_despeckle(img,goodImg):
    dsimg, homog = quickieHomo(img)
    return ul.filtered_match(img,dsimg,goodImg)

def detection(im):
   
    detector = cv2.SimpleBlobDetector_create()
    
    # Detect blobs.
    keypoints = detector.detect(im)
 
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)

def hist_first(original, *args, **kwargs):
    goodImg = cv2.imread('GoodImages\\3-A.PNG',0)
    length = 450
  
    img = np.copy(original)
    counts = np.zeros(length)
    diffs = np.zeros(length)
    index = [0]
    #plt.ion()
    systracker = np.zeros(2)
    img = ul.global_histogram(img,goodImg)
 
    newimg, systolic = despeckle_thresh(img, counts,diffs,index,systracker)
    return systolic

def despeckle_thresh(img, countarray,diffarray,index, systracker):
    dsimg, homog = quickHomog(img)
    homog2 = homog[80:200,200:400]
    #detection(homog)
    #thresholding(dsimg,homog2)
   
    #kernel = np.ones((3,3),np.uint8)
    #opening= cv2.morphologyEx(homog2,cv2.MORPH_OPEN,kernel, iterations = 1)

    edges = cv2.Canny(homog2, threshold1 = 50, threshold2 = 100)
  
    #cv2.imshow('edges', edges)

    #cv2.imshow('dialtion', opening)
   
    countarray[index[0]] = np.sum(edges)/255
    if index[0] > 5:
        diffarray[index[0]] = np.mean(countarray[index[0]-5:index[0]])
    else:
        diffarray[index[0]] = np.mean(countarray[:index[0]])

   

    systolic = countarray[index[0]] < diffarray[index[0]] 
    if systolic != systracker[0]:
        if index[0] - systracker[1] <= 10:
            systolic = not systolic
        else:
            systracker[0] = systolic
            systracker[1] = index[0]



    #print(systolic, countarray[index[0]], diffarray[index[0]])
    image =np.zeros((120,200,3), np.uint8)
    if systolic:
          image[:,:,1] = 255
        
    else:
          image[:,:,2] = 255

    index[0] = index[0] + 1

    #cv2.imshow('systolic',image)


    #plt.cla()
    #plt.plot(counts[:index[0]-1])
    #plt.title(video)
    #plt.pause(0.001)
        

    return dsimg, systolic


def thresholding(img,other):
    

    # otsu thresh
    
    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #thresh = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, -3)
    cv2.imshow('threshold',thresh)
   
    thresh = other
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)
    #closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 5)
    cv2.imshow('opening',opening)
    # sure background area
    #sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    #dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    #ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    #sure_fg = np.uint8(sure_fg)
    #unknown = cv2.subtract(sure_bg,sure_fg)
    #cv2.imshow('bg',sure_bg)
    #cv2.imshow('fg',sure_fg)
   # cv2.imshow('unknown',unknown)
    #cv2.imshow('dist',dist_transform)
    return thresh



def quickHomog(img):

     hScaler = 2

    
     wSize = 7

     hMat = np.zeros(img.shape)
     mean = cv2.blur(img.astype(np.float64),(wSize,wSize))
    
     #cv2.imshow('mean', mean)
     moment2 = cv2.blur(np.multiply(img,img).astype(np.float64), (wSize,wSize))

     dev = moment2-np.multiply(mean,mean)

     median = cv2.medianBlur(img,wSize)
     gaussian = cv2.GaussianBlur(img,(wSize,wSize),sigmaX=1)
    
     mean_mean = np.mean(mean) 
     mean_dev = np.mean(dev)

     #
     
     hthresh = np.ones(img.shape)*mean_dev**2/mean_mean/hScaler

     hVal = np.divide( np.multiply(dev,dev), mean)

     hMat = np.less_equal(hVal,hthresh)

     zeromean = np.less(mean,3)
     
     #hMat = np.multiply(hMat,np.logical_not(zeromean))

    
     hMat = np.logical_or(hMat,zeromean)

     gaussians = np.multiply(hMat,gaussian)
     medians = np.multiply(np.logical_not(hMat),median) 
     
     newimg = gaussians+medians


     #cv2.imshow('homogeny',hMat.astype(np.uint8)*255)
   
        
     return newimg.astype(np.uint8), hMat.astype(np.uint8)*255



def despeckle(img):
    winSize = 7 #nxn window for filtering
    halfWin = winSize/2

    #typical h for homogeneous region is 1.6 at window size 7x7
    # can play with this value to determine optimal threshold
    highThresh = 1

    sigmaX = 1
    sigmaY = 1

    pad = halfWin + 1 #how many pixels to pad the image


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
    for i in range(pad+1,size[0]-pad+1):
        for j in range(pad+1,size[1]-pad+1):
            W = img[i-halfWin:i+halfWin+1,j-halfWin:j+halfWin+1]
            mean = np.mean(W)
            vari = np.var(W)
            if mean == 0:
                h = 0
            else:
                h = vari/mean

            if h> highThresh: 
               # newimg[i,j] = np.median(W)
               pass
              
            else:
                #newimg[i,j] = np.sum(np.multiply(Gaussian,W))
                hMat[i,j] = 1

            #print(i,j, newimg[i,j])

    newimg = newimg.astype(np.uint8)
    newimg = newimg[pad+1:size[0]-pad+1, pad+1:size[1]-pad+1]
    cv2.imshow('despeckled', newimg)
    cv2.imshow('speckled', img)
    #plt.imshow(newimg, cmap='gray')
    #plt.xticks([])
    #plt.yticks([])
    #plt.show()


    cv2.imshow('homogeny', hMat)   


    #gimg = cv2.GaussianBlur(img, (7,7),sigmaX = 1);
    #cv2.imshow('gaussoan', gimg)

    #mimg = cv2.medianBlur(img, 7);
    #cv2.imshow('median', mimg)
    cv2.waitKey(33)

    return newimg.astype(np.uint8),hMat

if __name__ == "__main__":
    goodImg = cv2.imread('GoodImages\\3-A.png')
    vids = ['Videos/1-A.mp4', 'Videos/1-B.mp4', 'Videos/2-A.mp4', 'Videos/2-B.mp4',
        'Videos/3-A.mp4', 'Videos/3-B.mp4', 'Videos/4-A.mp4', 'Videos/4-B.mp4',
        'Videos/5-A.mp4', 'Videos/5-B.mp4', 'Videos/Varying.mp4']
    
    vids =['Videos/Varying.mp4']
    for video in vids:
        cap = cv2.VideoCapture(video)

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        counts = np.zeros(length)
        diffs = np.zeros(length)
        index = [0]
        #plt.ion()
        systracker = np.zeros(2)
        ul.runVideo(video, hist_first, counts,diffs,index, systracker)

        plt.plot(counts)
        mean = diffs
        dev = np.var(counts)**0.5
        hthresh = mean+dev/2
        lthresh = mean-dev/2


        plt.plot(mean)

        systolic = np.less(counts,mean).astype(np.uint8)
        systolic = systolic*(counts.max()-counts.min()) + counts.min()
        plt.plot(systolic)
        #plt.plot(hthresh)
        #plt.plot(lthresh)
        #plt.plot(diffs)
        plt.title(video)
        plt.show()



"""

# otsu thresh
#img = cv2.imread('GoodImages\\5-A.png')
#img = ul.stripFrame(img)
img = cv2.imread('Segmentation\\despeckled_3-A_h1.7.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('threshold',thresh)
cv2.waitKey(0)


# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('bg',sure_bg)
cv2.imshow('fg',sure_fg)
cv2.imshow('unknown',unknown)
cv2.imshow('dist',dist_transform)
cv2.waitKey(0)

"""

"""
#find and save despeckled image

speckImg = cv2.imread('GoodImages\\5-A.PNG',0)
speckImg = ul.stripFrame(speckImg)
despeckImg, hmat= quickieHomo(speckImg)
hmat = hmat*255
     
cv2.imshow('homogeny', hmat)


          
cv2.imshow('orig', speckImg)
cv2.imshow('despeck', despeckImg)
cv2.waitKey(0)
#cv2.imwrite('Segmentation\\despeckled_3-A_h1.png', despeckImg)

"""



"""
cap =  cv2.VideoCapture('Videos\\4-A.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out1 = cv2.VideoWriter('Videos\\despeckle_4-A.avi',fourcc, 5, (640,480))
out2 = cv2.VideoWriter('Videos\\homo_4-A.avi',fourcc, 5, (640,480))
ret = True
i = 1
while i <= 5:
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(i)
    i+=1
    if ret == True:
        frame = ul.stripFrame(frame)
        newframe,hFrame = despeckle(frame)
        out1.write(newframe)
        out2.write(newframe)
    else:
        break
cap.release()
out2.release()
out1.release()
cv2.destroyAllWindows()
"""

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




       
