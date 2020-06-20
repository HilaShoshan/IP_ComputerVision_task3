import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[y,x]...], [[dU,dV]...] for each points
    """
    pass


"""
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
"""


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    gauss_pyr = gaussianPyr(img, levels)
    for im in gauss_pyr:
        plt.gray()
        plt.imshow(im)
        plt.show()
    ans = [gauss_pyr[-1]]  # last level of laplacian pyramid is the smallest image in gaussianPyr
    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    g_kernel = cv2.getGaussianKernel(5, sigma)
    g_kernel = g_kernel.dot(g_kernel.T)
    i = levels - 1
    while i > 0:  # go through the list from end to start
        expand = gaussExpand(gauss_pyr[i], g_kernel)
        laplace = np.subtract(gauss_pyr[i-1], expand)
        ans.insert(0, laplace)
        i -= 1
    return ans


def build_laplacian_pyramid(im, max_levels = 4):
    """
    build a laplacian pyramid for the image
    :param im: ndarray
    :param max_levels: num of levels for the pyramid
    :param filter_size: size of filter for blurring
    :return: all levels of the pyramid and the filter vector
    """
    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    g_kernel = cv2.getGaussianKernel(5, sigma)
    g_kernel = g_kernel.dot(g_kernel.T)
    pyr = []
    gaussian = gaussianPyr(im, max_levels)
    for i in range(1, max_levels):
        temp = gaussExpand(gaussian[i], g_kernel)
        temp = gaussian[i-1] - temp
        pyr.append(temp)
    pyr.append(gaussian[max_levels - 1])
    return pyr


"""
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
"""


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:

    pass


"""
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
"""


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    ans = []
    width = (2**levels) * int(img.shape[1] / (2 ** levels))
    height = (2**levels) * int(img.shape[0] / (2 ** levels))
    img = cv2.resize(img, (width, height))  # resize the image to dimensions that can be divided into 2 x times
    ans.append(img)  # level 0 - the original image
    temp_img = img.copy()
    for i in range(1, levels):
        temp_img = reduce(temp_img)  # 2 times smaller image
        ans.append(temp_img)
    return ans


def gauss_blur(img: np.ndarray) -> np.ndarray:
    kernel_size = 5
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    g_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    g_kernel = g_kernel.dot(g_kernel.T)
    # g_kernel = cv2.getGaussianKernel(5, 0.3)  # sum of kernel = 1 (?)
    blur_img = cv2.filter2D(img, -1, g_kernel, borderType=cv2.BORDER_REPLICATE)
    return blur_img


def reduce(img: np.ndarray) -> np.ndarray:
    blur_img = gauss_blur(img)
    new_img = blur_img[::2, ::2]
    print("img: \n", blur_img[:10, :10], "\n new img: \n", new_img[:5, :5], '\n')
    return new_img


"""
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    expanded_img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
    # gs_k = 4 * (gs_k / np.sum(gs_k))  # make sure the sum of the kernel = 4
    blur_img = cv2.filter2D(expanded_img, -1, gs_k, borderType=cv2.BORDER_REPLICATE)
    return blur_img


"""
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
"""


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    naive_blend = naive_blending(img_1, img_2)

    return naive_blend


def naive_blending(img1: np.ndarray, img2: np.ndarray):
    pass


def super_naive_blending(img1: np.ndarray, img2: np.ndarray):
    width = img1.shape[1]
    height = img1.shape[0]
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (width, height))  # resize img2 to img1.shape
        print(img1.shape, img2.shape)
    blend_img = np.zeros(img1.shape)
    bound1 = int(width/3)
    bound2 = int(width*2/3)
    blend_img[:, :bound1] = img1[:, :bound1]  # take only the first image in the left piece of the image
    blend_img[:, bound2:] = img2[:, bound2:]  # take only the second image in the right piece of the image
    alpha = 1
    step = 1/(bound2-bound1)
    for i in range(bound1, bound2):  # gradually change the image
        blend_img[:, i] = alpha*img1[:, i] + (1-alpha)*img2[:, i]
        alpha -= step
    return blend_img


"""
def main():
    img_path = 'pyr_bit.jpg'
    img = cv2.imread(img_path, 0)
    list_gaus = gaussianPyr(img, 4)
    kernel_size = 5
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    list_exp = []
    for i in range(4):
        temp = gaussExpand(list_gaus[-1 - i], kernel)
        list_exp.append(temp)

    f, ax = plt.subplots(1, 4)
    ax[0].imshow(cv2.cvtColor(list_exp[0], cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(list_exp[1], cv2.COLOR_BGR2RGB))
    ax[2].imshow(cv2.cvtColor(list_exp[2], cv2.COLOR_BGR2RGB))
    ax[3].imshow(cv2.cvtColor(list_exp[3], cv2.COLOR_BGR2RGB))
    plt.show()
    """

def main():
        img_path = 'pyr_bit.jpg'
        img = cv2.imread(img_path, 0)
        list_gaus = gaussianPyr(img, 4)
        kernel_size = 5
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        for i in range(4):
            temp = gaussExpand(list_gaus[-1 - i], kernel)
            temp2 = cv2.pyrUp(list_gaus[-1 - i])
            if temp.all() == temp2.all():
                print("True")


if __name__ == '__main__':
    main()
