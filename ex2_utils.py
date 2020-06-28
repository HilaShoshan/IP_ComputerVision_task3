import time

import numpy as np
import cv2

def conv1D(in_signal: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param kernel_size: 1-D array as a kernel
    :return: The convolved array
    """

    if len(in_signal.shape) > 1:
        if in_signal.shape[1] > 1:
            raise ValueError("Input Signal is not a 1D array")
        else:
            in_signal = in_signal.reshape(in_signal.shape[0])

    inv_k = kernel_size[::-1].astype(np.float64)
    kernel_len = len(kernel_size)
    out_len = max(kernel_len, len(in_signal) + (kernel_len - 1))
    mid_kernel = kernel_len // 2
    padding = kernel_len - 1
    padded_signal = np.pad(in_signal, padding, 'constant')

    out_signal = np.ones(out_len)
    for i in range(out_len):
        st = i
        end = i + kernel_len

        out_signal[i] = (padded_signal[st:end] * inv_k).sum()

    return out_signal


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    img_h, img_w = in_image.shape[:2]
    kernel_shape = np.array([x for x in kernel.shape])
    mid_ker = kernel_shape // 2
    padded_signal = np.pad(in_image.astype(np.float32),
                           ((kernel_shape[0], kernel_shape[0]),
                            (kernel_shape[1], kernel_shape[1]))
                           , 'edge')

    out_signal = np.zeros_like(in_image)
    for i in range(img_h):
        for j in range(img_w):
            st_x = j + mid_ker[1] + 1
            end_x = st_x + kernel_shape[1]
            st_y = i + mid_ker[0] + 1
            end_y = st_y + kernel_shape[0]

            out_signal[i, j] = (padded_signal[st_y:end_y, st_x:end_x] * kernel).sum()

    return out_signal


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernel = np.array([[1, 0, -1]])
    x_drive = conv2D(in_image, kernel)
    y_drive = conv2D(in_image, kernel.T)

    ori = np.arctan2(y_drive, x_drive)
    mag = np.sqrt(x_drive ** 2 + y_drive ** 2)

    return ori, mag


def getSobelMagOri(img: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Extracts the Magnitude and Orientations of deriving the image using the Sobel kernel
    :param img: The Image
    :return: Magnitud, Orientation
    """
    sobel_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]).astype(np.float64)[::-1] / 8.0

    i_x = conv2D(img, sobel_kernel)
    i_y = conv2D(img, sobel_kernel.T)
    mag = np.hypot(i_x, i_y)
    ori = np.arctan2(i_y, i_x)
    return mag, ori


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    # CV solution
    cv_i_y = cv2.Sobel(img, -1, 0, 1, ksize=3)
    cv_i_x = cv2.Sobel(img, -1, 1, 0, ksize=3)
    cv_res = np.hypot(cv_i_x, cv_i_y)
    cv_res[cv_res < thresh] = 0
    cv_res[cv_res > 0] = 1

    # My solution
    mag, ori = getSobelMagOri(img)
    my_res = mag
    my_res[my_res < thresh] = 0
    my_res[my_res > 0] = 1
    return cv_res, my_res


def findZeroCrossing(lap_img: np.ndarray) -> np.ndarray:
    minLoG = cv2.morphologyEx(lap_img, cv2.MORPH_ERODE, np.ones((3, 3)))
    maxLoG = cv2.morphologyEx(lap_img, cv2.MORPH_DILATE, np.ones((3, 3)))
    zeroCross = np.logical_or(np.logical_and(minLoG < 0, lap_img > 0),
                              np.logical_and(maxLoG > 0, lap_img < 0))

    return zeroCross.astype(np.float)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    my_lap = cv2.Laplacian(img.astype(np.float), -1)
    return findZeroCrossing(my_lap)


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """
    # My implementation
    laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    my_gauss = createGaussianKernel(101)
    my_lap = cv2.filter2D(img.astype(np.float), -1, my_gauss)
    my_lap = cv2.filter2D(my_lap.astype(np.float), -1, laplacian)

    return findZeroCrossing(my_lap)


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    img = (img * 255).astype(np.float)
    st = time.time()
    cv_canny = cv2.Canny(img.astype(np.uint8), thrs_1, thrs_2)
    print("(Canny) CV Time:", time.time()-st)

    st = time.time()
    # My implementation
    # Getting Ix and Iy
    img_s = cv2.GaussianBlur(img, (5, 5), 1)
    img_s=img
    i_y = cv2.Sobel(img_s, -1, 0, 1, ksize=3)
    i_x = cv2.Sobel(img_s, -1, 1, 0, ksize=3)
    mag = np.hypot(i_x, i_y)
    ori = np.arctan2(i_y, i_x)

    # Degrees Quantisation
    quant_ori = np.degrees(ori)
    quant_ori[quant_ori < 0] += 180
    quant_ori = np.mod(((22.5 + quant_ori) // 45 * 45), 180)
    quant_ori[quant_ori > 135] = 0

    quant_ori_x = np.zeros_like(quant_ori, dtype=np.int)
    quant_ori_y = np.zeros_like(quant_ori, dtype=np.int)

    # angle 0
    quant_ori_x[quant_ori == 0] += 1
    # angle 45
    quant_ori_x[quant_ori == 45] -= 1
    quant_ori_y[quant_ori == 45] -= 1
    # angle 90
    quant_ori_y[quant_ori == 90] += 1
    # angle 135
    quant_ori_x[quant_ori == 135] += 1
    quant_ori_y[quant_ori == 135] -= 1

    h, w = img.shape[:2]
    pix_mag = mag[1:-1, 1:-1].reshape(-1)
    pix_y, pix_x = np.meshgrid(range(1, h - 1), range(1, w - 1))
    pix_x = pix_x.reshape(-1)
    pix_y = pix_y.reshape(-1)

    mag_sort = np.argsort(-pix_mag)
    pix_mag = pix_mag[mag_sort].tolist()
    pix_x = pix_x[mag_sort].tolist()
    pix_y = pix_y[mag_sort].tolist()

    # NMS
    mag_c = mag.copy()
    for max_idx in range(len(pix_mag)):
        ix = pix_x[max_idx]
        iy = pix_y[max_idx]

        grad_x = quant_ori_x[iy, ix]
        grad_y = quant_ori_y[iy, ix]
        v = mag[iy, ix]
        pre = mag[iy - grad_y,
                    ix - grad_x]
        post = mag[iy + grad_y,
                     ix + grad_x]

        if v <= pre or v <= post:
            mag_c[iy, ix] = 0

    # Hysteresis
    thrs_map_2 = ((mag_c >= thrs_2) & (mag_c <= thrs_1)).astype(np.uint8)
    thrs_map_1 = (mag_c >= thrs_1).astype(np.uint8)
    thrs_map_1_dilate = cv2.dilate(thrs_map_1, np.ones((2, 2)))

    my_canny = (thrs_map_1 | (thrs_map_2 & thrs_map_1_dilate))
    print("(Canny) My Time:", time.time()-st)
    return cv_canny, my_canny


def createGaussianKernel(k_size: int):
    if k_size % 2 == 0:
        raise ValueError("Kernel size should be an odd number")

    k = np.array([1, 1], dtype=np.float64)
    iter_v = np.array([1, 1], dtype=np.float64)

    for i in range(2, k_size):
        k = conv1D(k, iter_v)
    k = k.reshape((len(k), 1))
    kernel = k.dot(k.T)
    kernel = kernel / kernel.sum()
    return kernel


def blurImage1(inImage: np.ndarray, kernelSize: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    kernel = createGaussianKernel(kernelSize)

    return conv2D(inImage, kernel)


def blurImage2(inImage: np.ndarray, kernelSize: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """

    kernel = cv2.getGaussianKernel(kernelSize, -1)
    kernel = kernel.dot(kernel.T)
    blurred_img = cv2.filter2D(inImage, -1, kernel)
    return blurred_img


def nms(xyr: np.ndarray, radius: int) -> list:
    """
    Performes Non Maximum Suppression in order to remove circles that are close
    to each other to get a "clean" output.
    :param xyr:
    :param radius:
    :return:
    """
    ret_xyr = []

    while len(xyr) > 0:
        # Choose most ranked circle (MRC)
        curr_arg = xyr[:, -1].argmax()
        curr = xyr[curr_arg, :]
        ret_xyr.append(curr)
        xyr = np.delete(xyr, curr_arg, axis=0)

        # Find MRC close neighbors
        dists = np.sqrt(np.square(xyr[:, :2] - curr[:2]).sum(axis=1)) < radius
        idx_to_delete = np.where(dists)

        # Delete MRCs neighbors
        xyr = np.delete(xyr, idx_to_delete, axis=0)
    return ret_xyr


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """
    img = img.squeeze()
    if img.ndim > 2:
        raise ValueError("The image is not grayscale")

    h, w = img.shape
    max_radius = min(min(h, w) // 2, max_radius)

    # Get each pixels gradients direction
    i_y = cv2.Sobel(img, -1, 0, 1, ksize=3)
    i_x = cv2.Sobel(img, -1, 1, 0, ksize=3)
    ori = np.arctan2(i_y, i_x)

    # Get Edges using Canny Edge detector
    bw = cv2.Canny((img * 255).astype(np.uint8), 550, 100)

    radius_diff = max_radius - min_radius
    circle_hist = np.zeros((h, w, radius_diff))

    # Get the coordinates only for the edges
    ys, xs = np.where(bw)

    # Calculate the sin/cos for each edge pixel
    sins = np.sin(ori[ys, xs])
    coss = np.cos(ori[ys, xs])

    r_range = np.arange(min_radius, max_radius)
    for iy, ix, ss, cs in zip(ys, xs, sins, coss):
        grad_sin = (r_range * ss).astype(np.int)
        grad_cos = (r_range * cs).astype(np.int)

        xc_1 = ix + grad_cos
        yc_1 = iy + grad_sin

        xc_2 = ix - grad_cos
        yc_2 = iy - grad_sin

        # Check where are the centers that are in the image
        r_idx1 = np.logical_and(yc_1 > 0, xc_1 > 0)
        r_idx1 = np.logical_and(r_idx1, np.logical_and(yc_1 < h, xc_1 < w))

        # Check where are the centers that are in the image (Opposite direction)
        r_idx2 = np.logical_and(yc_2 > 0, xc_2 > 0)
        r_idx2 = np.logical_and(r_idx2, np.logical_and(yc_2 < h, xc_2 < w))

        # Add circles to the circle histogram
        circle_hist[yc_1[r_idx1], xc_1[r_idx1], r_idx1] += 1
        circle_hist[yc_2[r_idx2], xc_2[r_idx2], r_idx2] += 1

    # Find all the circles centers
    y, x, r = np.where(circle_hist > 11)
    circles = np.array([x, y, r + min_radius, circle_hist[y, x, r]]).T

    # Perform NMS
    circles = nms(circles, min_radius // 2)
    return circles
