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
    img = cv2.imread("lion.jpg", cv2.IMREAD_GRAYSCALE)
    print("original shape: ", img.shape)
    pyr_list = gaussianPyr(img)
    for im in pyr_list:
        print(im.shape)


def test_reduce():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    new_img = reduce(img)
    print(img.shape, '\n', img[:10, :10], '\n', "***************************", '\n')
    print(new_img.shape, '\n', new_img[:5, :5])


def test_gaussExpand():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    gs_k = np.array([1, 1])
    new_img = gaussExpand(img, gs_k)
    print(img.shape, '\n', img[:5, :5], '\n', "***************************", '\n')
    print(new_img.shape, '\n', new_img[:10, :10])


def main():
    # test_super_naive_blending()
    # test_gaussianPyr()
    # test_reduce()
    test_gaussExpand()


if __name__ == '__main__':
    main()