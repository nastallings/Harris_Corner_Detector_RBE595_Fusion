import numpy as np
import cv2


def RANSAC(k1, k2, matches, iterations=200, threshold=500):

    N = matches.shape[0]
    num_inliers = 0
    max_value = np.zeros(N)
    added_array = np.array([0, 0, 1])

    n_samples = int(N * 0.2)
    x1 = k1[matches[:, 0]]
    x2 = k2[matches[:, 1]]
    matched1 = np.hstack([x1, np.ones((x1.shape[0], 1))])
    matched2 = np.hstack([x2, np.ones((x2.shape[0], 1))])

    # start RANSAC
    for i in range(iterations):
        # Select Random Points
        list_max = np.zeros(N)
        idx = np.random.choice(N, n_samples, replace=False)
        p1 = matched1[idx, :]
        p2 = matched2[idx, :]

        # Perform Homography
        Homography_matrix = np.linalg.lstsq(p2, p1, rcond=None)[0]
        Homography_matrix[:, 2] = added_array

        # Compute Inlinears
        output = np.dot(matched2, Homography_matrix)
        list_max = np.linalg.norm(output - matched1, axis=1) ** 2 < threshold
        max_inlier_val = np.sum(list_max)

        if max_inlier_val > num_inliers:
            max_value = list_max.copy()
            num_inliers = max_inlier_val

    input = matched1[max_value]
    output = matched2[max_value]
    input_points = []
    output_points = []
    for index, point in enumerate(input):
        input_points.append((int(input[index][0]), int(input[index][1])))
        output_points.append((int(output[index][0]), int(output[index][1])))
        if index >= 3:
            break
    input_points = np.array(input_points)
    output_points = np.array(output_points)

    input_points = input_points.astype(np.float32)
    output_points = output_points.astype(np.float32)

    H = cv2.getPerspectiveTransform(input_points, output_points)

    return H, matches[max_value]
