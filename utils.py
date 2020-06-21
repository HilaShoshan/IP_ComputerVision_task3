import numpy as np
import scipy.ndimage as sp
from scipy import signal
import matplotlib.pyplot as plt
import cv2
import copy
from skimage.io import imread


def build_filter_vec(filter_size):
    """
    returns a row vector shape (1, filter_size)

    filter_vec = [[1]]
    conv = [[1, 1]]
    div = np.power(2, filter_size - 1)
    for i in range(filter_size - 1):
        filter_vec = signal.convolve2d(filter_vec, conv, mode="full")
    return filter_vec / div
    """
    sigma = 0.3 * ((filter_size - 1) * 0.5 - 1) + 0.8
    g_kernel = cv2.getGaussianKernel(filter_size, sigma)
    # g_kernel = g_kernel.dot(g_kernel.T)
    return g_kernel


def reduce_image(im):
    return im[::2, ::2]


def blur(im, filter_vec):
    """
    blur image
    :param im: an ndarray
    :param filter_vec: the filter for blurring
    :return: blurred image

    temp = sp.filters.convolve(im, filter_vec)
    return sp.filters.convolve(temp, np.transpose(filter_vec))
    """
    temp = cv2.filter2D(im, -1, filter_vec, borderType=cv2.BORDER_REPLICATE)
    return cv2.filter2D(temp, -1, np.transpose(filter_vec), borderType=cv2.BORDER_REPLICATE)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    builds guassian pyramid out of the image
    :param im: ndarray
    :param max_levels: num of levels for the pyramid
    :param filter_size: the size of filter for blurring
    :return: all levels of the pyramid and the filter vector
    """
    filter_vec = build_filter_vec(filter_size)
    pyr = [im]
    for i in range(1, max_levels):
        temp = blur(pyr[i-1], filter_vec)
        pyr.append(reduce_image(temp))
    return pyr, filter_vec


def zero_pad(im):
    """
    pad an image with zeros
    :param im: ndarray
    :return: padded image
    """
    padded_im = np.zeros((im.shape[0]*2, im.shape[1]*2))
    padded_im[::2, ::2] = im
    return padded_im


def expand(im, filter_vec):
    """
    expand an image by 2
    :param im: ndarray
    :param filter_vec: filter for blurring
    :return:
    """
    padded_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    padded_im[::2, ::2] = im
    return blur(padded_im, 2 * filter_vec)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    build a laplacian pyramid for the image
    :param im: ndarray
    :param max_levels: num of levels for the pyramid
    :param filter_size: size of filter for blurring
    :return: all levels of the pyramid and the filter vector
    """
    filter_vec = build_filter_vec(filter_size)
    pyr = []
    gaussian = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    for i in range(1, max_levels):
        temp = expand(gaussian[i], filter_vec)
        temp = gaussian[i-1] - temp
        pyr.append(temp)
    pyr.append(gaussian[max_levels - 1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    reconstructs image from laplacian pyramid
    :param lpyr: laplacian pyramid
    :param filter_vec: filter vector
    :param coeff: size of the level of the pyramid. multiply each level of pyramid by corresponding coeff
    :return: image
    """
    lpyr = lpyr * np.array(coeff)
    im = lpyr[-1]
    for i in range(len(lpyr) - 2, -1, -1):
        im = expand(im, filter_vec) + lpyr[i]
    return im


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: are two input grayscale images to be blended.
    :param im2: are two input grayscale images to be blended.
    :param mask: is a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
            of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
            and False corresponds to 0.
    :param max_levels: is the max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im:  is the size of the Gaussian filter (an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar that represents a squared filter) which
        defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: blended image from the 2 images.
    """
    L1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    mask_pyr = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]
    Lout = []
    for i in range(max_levels):
        Lout.append(L1[i]*mask_pyr[i] + (1-mask_pyr[i]) * L2[i])
    coeff = [1] * max_levels
    return laplacian_to_image(Lout, filter_vec, coeff)


def main():
    img1 = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("lion.jpg", cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread("mask_cat.jpg", cv2.IMREAD_GRAYSCALE)
    ans = pyramid_blending(img1, img2, mask, 5, 5, 5)
    plt.gray()
    plt.imshow(ans)
    plt.show()


if __name__ == '__main__':
    main()
