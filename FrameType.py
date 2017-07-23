import numpy as np

def f_type(frame):

    frameMean = np.mean(frame)

    if 5 <= frameMean <= 5.99:
        return '1'
    elif 6 <= frameMean <= 7.99:
        return '2'
    elif 8 <= frameMean <= 15.99:
        return '3'
    elif 16 <= frameMean <= 64.99:
        return '4'
    elif 65 <= frameMean <= 115:
        return '5'
