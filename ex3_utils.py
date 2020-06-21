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


def get_gaussian1D():
    kernel_size = 5
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    g_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    return g_kernel


def gauss_blur(img: np.ndarray, filter_vec) -> np.ndarray:
    temp = cv2.filter2D(img, -1, filter_vec, borderType=cv2.BORDER_REPLICATE)
    return cv2.filter2D(temp, -1, np.transpose(filter_vec), borderType=cv2.BORDER_REPLICATE)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    gauss_pyr = gaussianPyr(img, levels)
    ans = [gauss_pyr[-1]]  # last level of laplacian pyramid is the smallest image in gaussianPyr
    g_kernel = get_gaussian1D()
    i = levels - 1
    while i > 0:  # go through the list from end to start
        expand = gaussExpand(gauss_pyr[i], g_kernel)
        laplace = np.subtract(gauss_pyr[i-1], expand)
        ans.insert(0, laplace)
        i -= 1
    return ans


"""
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
"""


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    levels = len(lap_pyr)
    temp = lap_pyr[-1]  # the smallest image (from the gaussPyramid)
    gs_k = get_gaussian1D()
    i = levels - 1
    while i > 0:  # go through the list from end to start
        expand = gaussExpand(temp, gs_k)
        temp = expand + lap_pyr[i - 1]
        i -= 1
    return temp


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


def reduce(img: np.ndarray) -> np.ndarray:
    g_kernel = get_gaussian1D()
    blur_img = gauss_blur(img, g_kernel)
    new_img = blur_img[::2, ::2]
    return new_img


"""
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    padded_im = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    padded_im[::2, ::2] = img
    return gauss_blur(padded_im, 2 * gs_k)


"""
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
"""


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    img_1, img_2 = resize_as_mask(img_1, img_2, mask)
    naive_blend = naive_blending(img_1, img_2, mask)
    l_a = laplaceianReduce(img_1, levels)
    l_b = laplaceianReduce(img_2, levels)
    g_m = gaussianPyr(mask, levels)
    l_c = []  # new laplacian pyramid
    for k in range(levels):
        """"
        height = l_a[k].shape[0]
        width = l_a[k].shape[1]
        new_img = np.zeros(l_a[k].shape)
        for i in range(height):
            for j in range(width):
                new_img[i, j] = g_m[k][i, j] * l_a[k][i, j] + (1 - g_m[k][i, j]) * l_b[k][i, j]
        l_c.append(new_img)
        """""
        l_c.append(l_a[k] * g_m[k] + (1 - g_m[k]) * l_b[k])
    pyr_blend = laplaceianExpand(l_c)  # np.clip(laplaceianExpand(l_c), 0, 1)
    return naive_blend, pyr_blend


def resize_as_mask(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> (np.ndarray, np.ndarray):
    new_width = mask.shape[1]
    new_height = mask.shape[0]
    img1 = cv2.resize(img1, (new_width, new_height))
    img2 = cv2.resize(img2, (new_width, new_height))
    return img1, img2


def naive_blending(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ans = np.empty_like(img1.shape)
    is_mask_1 = mask == 1
    for i in range(mask.shape[0]):  # rows
        for j in range(mask.shape[1]):  # cols
            if is_mask_1[i, j]:
                ans[i, j] = img1[i, j]
            else:
                ans[i, j] = img2[i, j]
    #ans[mask == 1] = img1[mask == 1]
    #ans[mask == 0] = img2[mask == 0]
    return ans


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
