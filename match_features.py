import cv2
import numpy as np
from cv2 import GaussianBlur
from scipy.spatial.distance import cdist


def match_features (image_1, keyPoints_1, image_2, keyPoints_2):
    window = 5
    descriptor1 = []
    descriptor2 = []

    for i, kp1 in enumerate(keyPoints_1):
        y_1, x_2 = kp1
        y_1 = int(y_1)
        x_2 = int(x_2)
        patch_1 = image_1[y_1 - int(window / 2):y_1 + int((window + 1) / 2), x_2 - int(window / 2):x_2 + int((window + 1) / 2)]
        descriptor1.append(create_descriptor(patch_1))

    for j, kp2 in enumerate(keyPoints_2):
        y_2, x_2 = kp2
        y_2 = int(y_2)
        x_2 = int(x_2)
        patch_2 = image_2[y_2 - int(window / 2):y_2 + int((window + 1) / 2), x_2 - int(window / 2):x_2 + int((window + 1)/ 2)]
        descriptor2.append(create_descriptor(patch_2))

    return find_close_descriptors(np.array(descriptor1), np.array(descriptor2))


def create_descriptor(window):
    std = np.std(window)
    if std == 0:
        std = 1
    normalize = (window-np.mean(window)) / std
    return normalize.flatten()


def find_close_descriptors(descriptor1, descriptor2):
    threshold = .9
    pairs = []
    dists = cdist(descriptor1, descriptor2, 'euclidean')
    for i in range(descriptor1.shape[0]):
        min_array = np.sort(dists[i, :])
        if min_array[1] != 0:
            if (min_array[0] / min_array[1]) <= threshold:
                pairs.append([i, np.argmin(dists[i, :])])

    return np.asarray(pairs)


