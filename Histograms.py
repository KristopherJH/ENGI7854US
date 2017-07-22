import cv2

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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
    s_values_2= s_values[20:]
    s_counts_2= s_counts[20:]

    t_values, t_counts = np.unique(template, return_counts=True)

    t_values_2 = t_values[20:]
    t_counts_2 = t_counts[20:]


    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts_2).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts_2).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values_2 = np.interp(s_quantiles, t_quantiles, t_values_2)

    interp_t_values = s_values
    interp_t_values[0:20] = s_values[0:20]
    interp_t_values[20:] = interp_t_values_2

    return np.floor(interp_t_values[bin_idx].reshape(oldshape))



goodImg = cv2.imread('Images\\3-A_frame_39.png',0)
badImg = cv2.imread('Images\\5-A_frame_20.png',0)


matchedImg = hist_match(badImg,goodImg)
matchedImg = matchedImg.astype(np.uint8)


# create a CLAHE object (Arguments are optional).
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl1 = clahe.apply(matchedImg)

cv2.imshow('Hello World', matchedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()










