import numpy as np
import cv2
import matplotlib.pyplot as plt

from ex3_utils import *


def test_naive_blending():
    img1 = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("lion.jpg", cv2.IMREAD_GRAYSCALE)
    blend = naive_blending(img1, img2)
    plt.gray()
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
    plt.imshow(blend)
    plt.show()


def main():
    test_naive_blending()


if __name__ == '__main__':
    main()