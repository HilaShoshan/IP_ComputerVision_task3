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

    return np.zeros(im1.shape), np.zeros(im1.shape)


from scipy import signal


def optical_flow(I1g, I2g, window_size=5, tau=10):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    w = int(window_size / 2)  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)

    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0] - w):
        for j in range(w, I1g.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
            # b = ... # get b here
            # A = ... # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            nu = ...  # get velocity here
            u[i, j] = nu[0]
            v[i, j] = nu[1]

    return u, v


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
    ans = np.zeros(img1.shape)
    is_mask_1 = mask == 1
    for i in range(mask.shape[0]):  # rows
        for j in range(mask.shape[1]):  # cols
            if is_mask_1[i, j]:
                ans[i, j] = img1[i, j]
            else:
                ans[i, j] = img2[i, j]
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
