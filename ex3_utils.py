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

"""""
def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    kernel = np.array([[1, 0, -1]])
    x_drive = cv2.filter2D(im2, -1, kernel)
    y_drive = cv2.filter2D(im2, -1, kernel.T)
    t_drive = im2 - im1
    lamda_1, lamda_2 = np.linalg.eigvals(ATA)
    for i in range(win_size, im1.shape[0] - win_size + 1, step_size):
        for j in range(win_size, im1.shape[1] - win_size + 1, step_size):
            x = x_drive[i-int(win_size/2):i+int(win_size/2), j-int(win_size/2):j+int(win_size/2)]
"""""

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    kernel = np.array([[-1, 0, 1]])
    x_drive = cv2.filter2D(im2, -1, kernel)
    y_drive = cv2.filter2D(im2, -1, kernel.T)
    t_drive = im2 - im1
    points = []
    u_v = []
    for i in range(win_size, im1.shape[0] - win_size + 1, step_size):
        for j in range(win_size, im1.shape[1] - win_size + 1, step_size):
            x = x_drive[i - int(win_size / 2):i + int(win_size / 2), j - int(win_size / 2):j + int(win_size / 2)]
            y = y_drive[i - int(win_size / 2):i + int(win_size / 2), j - int(win_size / 2):j + int(win_size / 2)]
            t = t_drive[i - int(win_size / 2):i + int(win_size / 2), j - int(win_size / 2):j + int(win_size / 2)]
            AtA = [[(x*x).sum(), (x*y).sum()],
                   [(x*y).sum(), (y*y).sum()]]
            lamdas = np.linalg.eigvals(AtA)
            lamda2 = np.min(lamdas)
            lamda1 = np.max(lamdas)
            if lamda2 <= 1 or lamda1/lamda2 >= 100:
                continue
            arr = [[-(x*t).sum()], [-(y*t).sum()]]
            u_v.append(np.linalg.inv(AtA).dot(arr))
            points.append([j, i])
    return np.array(points), np.array(u_v)


"""
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
"""


def get_gaussian():
    kernel = cv2.getGaussianKernel(5, -1)
    kernel = kernel.dot(kernel.T)
    return kernel


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    gauss_pyr = gaussianPyr(img, levels)
    g_kernel = get_gaussian() * 4
    for i in range(levels - 1):
        gauss_pyr[i] = gauss_pyr[i] - gaussExpand(gauss_pyr[i+1], g_kernel)
    return gauss_pyr


"""
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
"""


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    levels = len(lap_pyr)
    temp = lap_pyr[-1]  # the smallest image (from the gaussPyramid)
    gs_k = get_gaussian() * 4
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
    img = img.astype(np.float64)
    ans.append(img)  # level 0 - the original image
    temp_img = img.copy()
    for i in range(1, levels):
        temp_img = reduce(temp_img)  # 2 times smaller image
        ans.append(temp_img)
    return ans


def reduce(img: np.ndarray) -> np.ndarray:
    g_kernel = get_gaussian()
    blur_img = cv2.filter2D(img, -1, g_kernel)
    new_img = blur_img[::2, ::2]
    return new_img


"""
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    if img.ndim == 3:  # RGB img
        padded_im = np.zeros(((img.shape[0] * 2), (img.shape[1] * 2), 3))
    else:
        padded_im = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    padded_im[::2, ::2] = img
    return cv2.filter2D(padded_im, -1, gs_k)


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
    ans = (l_a[levels-1] * g_m[levels-1] + (1 - g_m[levels-1]) * l_b[levels-1])
    gs_k = get_gaussian() * 4
    k = levels - 2
    while k >= 0:  # go through the list from end to start
        ans = gaussExpand(ans, gs_k) + l_a[k] * g_m[k] + (1 - g_m[k]) * l_b[k]
        k -= 1
    naive_blend = cv2.resize(naive_blend, (ans.shape[1], ans.shape[0]))
    return naive_blend, ans


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


