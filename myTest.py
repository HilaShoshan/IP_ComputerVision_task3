import numpy as np
import cv2
import matplotlib.pyplot as plt

from ex3_utils import *


def test_super_naive_blending():
    img1 = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("lion.jpg", cv2.IMREAD_GRAYSCALE)
    blend = super_naive_blending(img1, img2)
    plt.gray()
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
    plt.imshow(blend)
    plt.show()


def test_gaussianPyr():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    print("original shape: ", img.shape)
    pyr_list = gaussianPyr(img)
    for im in pyr_list:
        print(im.shape)
        plt.gray()
        plt.imshow(im)
        plt.show()


def test_reduce():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    new_img = reduce(img)
    print(img.shape, '\n', img[:10, :10], '\n', "***************************", '\n')
    print(new_img.shape, '\n', new_img[:5, :5])


def test_gaussExpand():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    kernel = cv2.getGaussianKernel(5, 0.3)
    gs_k = kernel.transpose() * kernel
    print(np.sum(gs_k))
    new_img = gaussExpand(img, gs_k)
    print(img.shape, '\n', img[:5, :5], '\n', "***************************", '\n')
    print(new_img.shape, '\n', new_img[:10, :10])
    plt.gray()
    plt.imshow(img)
    plt.show()
    plt.imshow(new_img)
    plt.show()


def gaussianPyrAndExpand():
    img = cv2.cvtColor(cv2.imread("cat.jpg"), cv2.COLOR_BGR2RGB)
    plt.gray()
    print("original shape: ", img.shape)
    pyr_list = gaussianPyr(img)
    for im in pyr_list:
        print(im.shape)
        plt.imshow(im)
        plt.show()
    kernel_size = 5
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    gs_k = cv2.getGaussianKernel(kernel_size, sigma)
    expand = gaussExpand(pyr_list[1], gs_k)
    plt.imshow(expand)
    plt.show()


def test_laplaceianReduce():
    img = cv2.imread("pyr_bit.jpg", cv2.IMREAD_GRAYSCALE)
    laplacePyr = laplaceianReduce(img, 6)
    for im in laplacePyr:
        print(im.shape)
        plt.gray()
        plt.imshow(im)
        plt.show()


def test_laplaceianExpand():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    lap_pyr = laplaceianReduce(img)
    orig = laplaceianExpand(lap_pyr)
    plt.gray()
    plt.imshow(orig)
    plt.show()
    plt.imshow(img)
    plt.show()


def test_pyrBlend():
    img2 = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread("sunset.jpg", cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread("mask_cat.jpg", cv2.IMREAD_GRAYSCALE)
    naive, pyr = pyrBlend(img1, img2, mask, 5)
    plt.gray()
    plt.imshow(pyr)
    plt.show()
    plt.imshow(naive)
    plt.show()


def main():
    # test_super_naive_blending()
    # test_gaussianPyr()
    # test_reduce()
    # test_gaussExpand()
    # gaussianPyrAndExpand()
    # test_laplaceianReduce()
    # test_laplaceianExpand()
    test_pyrBlend()


if __name__ == '__main__':
    main()