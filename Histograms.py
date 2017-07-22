import cv2

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def stripFrame(frame):
    return frame[33:411, 98:583]

def adaptive_histMatch(source, template, neighbourhood = 50):


    area = source.shape
    finalSource = np.zeros(area)
    numX = np.ceil(area[0]/neighbourhood).astype(int)
    numY = np.ceil(area[1]/neighbourhood).astype(int)
    for i in range(area[0]):
        startX = i 
        if startX+neighbourhood >= area[0]:
                startX = area[0] - neighbourhood
        for j in range(area[1]):
            startY = j
            if startY+neighbourhood >= area[1]:
                startY = area[1] - neighbourhood
            
            
            tempSource = source[startX:startX+neighbourhood, startY:startY+neighbourhood]
            tempTemplate = template[startX:startX+neighbourhood, startY:startY+neighbourhood]

            newsource = hist_match(tempSource,tempTemplate)

            source[startX:startX+neighbourhood, startY:startY+neighbourhood] = newsource

    return source




def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    #s_values_2= s_values[20:]
    #s_counts_2= s_counts[20:]

    t_values, t_counts = np.unique(template, return_counts=True)

    #t_values_2 = t_values[20:]
    #t_counts_2 = t_counts[20:]


    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

   # interp_t_values = s_values
    #interp_t_values[0:20] = s_values[0:20]
    #interp_t_values[20:] = interp_t_values_2

    return np.floor(interp_t_values[bin_idx].reshape(oldshape))

def adaptive_histogram(badImg):

      #matchedImg = adaptive_histMatch(badImg,goodImg)
    goodImg = stripFrame(cv2.imread('GoodImages\\3-A.png',0))
    matchedImg = hist_match(badImg,goodImg)
    matchedImg = matchedImg.astype(np.uint8)
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(matchedImg)

def global_histogram(badImg):
     #matchedImg = adaptive_histMatch(badImg,goodImg)
    goodImg = stripFrame(cv2.imread('GoodImages\\3-A.png',0))
    matchedImg = hist_match(badImg,goodImg)
    return matchedImg.astype(np.uint8)
    



def runVideo(video,funcToUse):


    cap = cv2.VideoCapture(video)
    


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

            editedImg = funcToUse(badImg)

            cv2.imshow("Good", editedImg)
            cv2.imshow("Bad", badImg)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


runVideo('Videos/1-A.mp4', global_histogram)










