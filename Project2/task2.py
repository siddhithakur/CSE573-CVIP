"""
 Grayscale Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with two commonly used 
image processing techniques: image denoising and edge detection. 
Specifically, you are given a grayscale image with salt-and-pepper noise, 
which is named 'task2.png' for your code testing. 
Note that different image might be used when grading your code. 

You are required to write programs to: 
(i) denoise the image using 3x3 median filter;
(ii) detect edges in the denoised image along both x and y directions using Sobel operators (provided in line 30-32).
(iii) design two 3x3 kernels and detect edges in the denoised image along both 45° and 135° diagonal directions.
Hint: 
• Zero-padding is needed before filtering or convolution. 
• Normalization is needed before saving edge images. You can normalize image using the following equation:
    normalized_img = 255 * frac{img - min(img)}{max(img) - min(img)}

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy for basic matrix calculations EXCEPT any function/operation related to convolution or correlation. 
You should NOT use any other libraries, which provide APIs for convolution/correlation ormedian filtering. 
Please write the convolution code ON YOUR OWN. 
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)

sobel_45=np.array(([[2,1, 0], [1, 0, -1], [0, -1, -2]])).astype(int)
sobel_135=np.array(([[0,-1,-2],[1,0,-1],[2,1,0]])).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """

    # TO DO: implement your solution here
    # raise NotImplementedError
    # print(img.shape)
    img = np.pad(img, ((1, 1), (1, 1)), 'constant')

    denoise_img = np.zeros((img.shape[0] - 2, img.shape[1] - 2))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            s = img[i-1:i + 2, j-1:j + 2]
            denoise_img[i-1, j-1] = np.median(s)
    denoise_img=denoise_img.astype(int)
    # print(denoise_img.shape)
    return denoise_img



def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """

    # TO DO: implement your solution here
    # raise NotImplementedError
    kernel = np.flip(kernel)
    size=sobel_x.shape[0]
    conv_img = np.zeros((img.shape[0]-2, img.shape[1]-2))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            s = img[i-1:i + size-1, j-1:j + size-1]
            conv_img[i -1,j-1] = np.sum(np.multiply(s, kernel))

    return conv_img

def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image, 
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    # raise NotImplementedError

    img = np.pad(img, ((1, 1),(1,1)), 'constant')

    x = convolve2d(img, sobel_x)
    y = convolve2d(img, sobel_y)
    # print(x)
    em = (np.sqrt((x**2)+(y**2))).astype(int)
    # print(edge_mag)

    denom = (np.max(em) - np.min(em))
    num = em - np.min(em)
    edge_mag = ((255 * num) / denom).astype(int)
    denom =  (np.max(x) - np.min(x))
    num = x - np.min(x)
    edge_x = ((255 * num)/denom).astype(int)
    denomy = (np.max(y) - np.min(y))
    numy = y - np.min(y)
    edge_y = ((255 * numy) / denomy).astype(int)

    # print(edge_x)
    # print(edge_x.shape,edge_y.shape,edge_mag.shape)


    return edge_x, edge_y, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    #raise NotImplementedError

    img = np.pad(img, ((1, 1), (1, 1)), 'constant')

    x = convolve2d(img, sobel_45)
    y = convolve2d(img, sobel_135)


    denom = (np.max(x) - np.min(x))
    num = x - np.min(x)
    edge_45 = ((255 * num) / denom).astype(int)
    denomy = (np.max(y) - np.min(y))
    numy = y - np.min(y)
    edge_135 = ((255 * numy) / denomy).astype(int)
    # print(edge_135)
    # print(edge_45.shape, edge_135.shape)
    print(sobel_45,sobel_135) # print the two kernels you designed here
    return edge_45, edge_135


if __name__ == "__main__":
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    imwrite('results/task2_denoise.jpg', denoise_img)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    imwrite('results/task2_edge_x.jpg', edge_x_img)
    imwrite('results/task2_edge_y.jpg', edge_y_img)
    imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    imwrite('results/task2_edge_diag2.jpg', edge_135_img)





