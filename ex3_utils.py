import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


"""
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
"""


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    Ix = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=5)
    It = im2 - im1
    points = []
    d = []
    for i in range(win_size, im1.shape[0] - win_size + 1, step_size):
        for j in range(win_size, im1.shape[1] - win_size + 1, step_size):
            starti, startj, endi, endj = i - win_size // 2, j - win_size // 2, i + win_size // 2 + 1, j + win_size // 2 + 1
            b = -(It[starti:endi, startj:endj]).reshape(win_size ** 2, 1)
            A = np.asmatrix(np.concatenate((Ix[starti:endi, startj:endj].reshape(win_size ** 2, 1),
                                            Iy[starti:endi, startj:endj].reshape(win_size ** 2, 1)), axis=1))
            values, vec = np.linalg.eig(A.T * A)
            values.sort()
            values = values[::-1]
            if values[0] >= values[1] > 1 and values[0] / values[1] < 100:
                # v = (A.T * A).I * A.T * b
                v = np.array(np.dot(np.linalg.pinv(A), b))
                points.append(np.array([j, i]))
                d.append(v[::-1].copy())
    return np.array(points), np.array(d)


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
    g_kernel = get_gaussian1D() * 4
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
    if img.shape[-1] == 3:  # RGB img
        padded_im =  np.zeros(((img.shape[0] * 2), (img.shape[1] * 2), 3))
        padded_im[::2, ::2, :] = img
    else:
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
    l_c = [None] * levels  # new laplacian pyramid
    l_c[-1] = (l_a[levels-1] * g_m[levels-1] + (1 - g_m[levels-1]) * l_b[levels-1])
    gs_k = get_gaussian1D() * 4
    k = levels - 2
    while k >= 0:  # go through the list from end to start
        x = gaussExpand(l_c[k+1], gs_k) + l_a[k] * g_m[k] + (1 - g_m[k]) * l_b[k]
        l_c[k] = x
        k -= 1
    pyr_blend = laplaceianExpand(l_c)  # np.clip(laplaceianExpand(l_c), 0, 1)
    new_width = pyr_blend.shape[1]
    new_height = pyr_blend.shape[0]
    naive_blend = cv2.resize(naive_blend, (new_width, new_height))
    return naive_blend, pyr_blend


def resize_as_mask(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> (np.ndarray, np.ndarray):
    new_width = mask.shape[1]
    new_height = mask.shape[0]
    img1 = cv2.resize(img1, (new_width, new_height))
    img2 = cv2.resize(img2, (new_width, new_height))
    return img1, img2


def naive_blending(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ans = img1*mask + img2*(1-mask)
    return ans


def super_naive_blending(img1: np.ndarray, img2: np.ndarray):
    width = img1.shape[1]
    height = img1.shape[0]
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (width, height))  # resize img2 to img1.shape
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
