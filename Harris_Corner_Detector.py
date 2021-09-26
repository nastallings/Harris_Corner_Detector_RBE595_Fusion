import numpy as np
import matplotlib.pyplot as plt
import cv2


def sobel_operation(x_filter, y_filter, img, k):
    rows, columns = img.shape
    filter_image_x = np.zeros(shape=(rows, columns))
    filter_image_y = np.zeros(shape=(rows, columns))
    for i in range(rows-2):
        for j in range(columns-2):
            filter_image_x[i + 1, j + 1] = np.sum(np.multiply(x_filter, img[i:i+k, j:j+k]))
            filter_image_y[i + 1, j + 1] = np.sum(np.multiply(y_filter, img[i:i+k, j:j+k]))
    return filter_image_x, filter_image_y


def get_highest_peaks(list_of_corners):
    threshold = .10
    x, y = list_of_corners.shape
    output_values = np.zeros((x*y, 2))
    max_value = np.max(np.amax(list_of_corners, axis=0))
    index = 0
    for i in range(0, x):
        if list_of_corners[i][2] > max_value * threshold:
            output_values[index, :] = [list_of_corners[i][0], list_of_corners[i][1]]
            index += 1

    return output_values[0:index, :], index


def harris_corner_detector(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    window_shape = 3

    x, y = img.shape
    offset = int(window_shape/2)
    window = np.ones((window_shape, window_shape))
    response = np.zeros((x*y, 3))

    Sobel_X = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Sobel_Y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    Ix, Iy = sobel_operation(Sobel_X, Sobel_Y, img, 3)
    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    Ixy = Ix * Iy

    index = 0
    for i in range(offset, x - offset):
        for j in range(offset, y - offset):
            x_offset = i - offset
            x_wall = i + offset + 1

            y_offset = j - offset
            y_wall = j + offset + 1

            Sum_Ixx = np.sum(window * Ixx[x_offset:x_wall, y_offset:y_wall])
            Sum_Iyy = np.sum(window * Iyy[x_offset:x_wall, y_offset:y_wall])
            Sum_Ixy = np.sum(window * (Ixy[x_offset:x_wall, y_offset:y_wall]))

            matrix = np.array([[Sum_Ixx, Sum_Ixy],
                               [Sum_Ixy, Sum_Iyy]])
            alpha = .04
            M = np.linalg.det(matrix) - (alpha * ((np.trace(matrix)) ** 2))
            if M > 0:
                response[index, :] = [i, j, M]
                index += 1
    return get_highest_peaks(response[0:index+1, :])

def plot_harris_points(harris_output, img):
    plt.imshow(img)
    plt.plot(harris_output[:, 1], harris_output[:, 0], 'ro')
    plt.show()


