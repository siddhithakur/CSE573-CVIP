"""
Morphology Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with commonly used morphology
binary image processing techniques. Use the proper combination of the four commonly used morphology operations, 
i.e. erosion, dilation, open and close, to remove noises and extract boundary of a binary image. 
Specifically, you are given a binary image with noises for your testing, which is named 'task3.png'.  
Note that different binary image might be used when grading your code. 

You are required to write programs to: 
(i) implement four commonly used morphology operations: erosion, dilation, open and close. 
    The stucturing element (SE) should be a 3x3 square of all 1's for all the operations.
(ii) remove noises in task3.png using proper combination of the above morphology operations. 
(iii) extract the boundaries of the objects in denoised binary image 
      using proper combination of the above morphology operations. 
Hint: 
• Zero-padding is needed before morphology operations. 

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy libraries, HOWEVER, 
you are NOT allowed to use any functions or APIs directly related to morphology operations.
Please implement erosion, dilation, open and close operations ON YOUR OWN.
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np


def morph_erode(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return erode_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology erosion on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    img=img.astype(int)
    structuring_element=np.ones((3,3),dtype=int)

    erode_img = np.zeros((img.shape[0], img.shape[1]),dtype=int)


    img = np.pad(img, ((1, 1), (1, 1)), 'constant')
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            s = img[i-1:i + 2, j-1:j + 2]
            eroding=(s==structuring_element)
            if np.all(eroding==True):
                erode_img[i-1,j-1]=1
            else:
                erode_img[i-1,j-1]=0
    # imshow('eorde',erode_img)
    # waitKey(0)

    return erode_img


def morph_dilate(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    img = img.astype(int)
    structuring_element = np.ones((3, 3), dtype=int)

    dilate_img = np.zeros((img.shape[0], img.shape[1]),dtype=int)


    img = np.pad(img, ((1, 1), (1, 1)), 'constant')
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            s = img[i - 1:i + 2, j - 1:j + 2]
            dilating = (s == structuring_element)
            if np.any(dilating == True):
                dilate_img[i - 1, j - 1] = 1
            else:
                dilate_img[i - 1, j - 1] = 0

    return dilate_img


def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """

    # TO DO: implement your solution here
    # raise NotImplementedError
    eroded = morph_erode(img)
    open_img = morph_dilate(eroded)
    return open_img


def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """

    # TO DO: implement your solution here
    # raise NotImplementedError
    dilated_img=morph_dilate(img)
    close_img=morph_erode(dilated_img)
    return close_img



def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    img=np.array(img,dtype=bool)
    # morph_erode(img)
    # morph_dilate(img)
    opened=morph_open(img)

    denoise_img=morph_close(opened)
    # imshow('denoise', denoise_img)
    # waitKey(0)

    denoise_img[denoise_img == 1] = 255
    denoise_img[denoise_img == 0] = 0
    return denoise_img


def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    img = np.array(img, dtype=bool)
    eorded_img=morph_erode(img)
    dilated_img=morph_dilate(img)
    bound_img=img.astype(int)-eorded_img
    bound_img[bound_img == 1] = 255
    return bound_img


if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)





