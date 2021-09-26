import matplotlib.pyplot as plt
import numpy as np
import cv2
import Harris_Corner_Detector as HCD
from RANSAC import RANSAC
import match_features as mf

# Load Images
# image_1 = cv2.imread("scene_2.jpeg")
# image_2 = cv2.imread("scene_3.jpeg")
# image_3 = cv2.imread("scene_1.jpeg")
# image_4 = cv2.imread("scene_4.jpeg")
#
# # Detect corners using Harris
# output_1, num_points_1 = HCD.harris_corner_detector(image_1)
#
# output_2, num_points_2 = HCD.harris_corner_detector(image_2)
#
# output_3, num_points_3 = HCD.harris_corner_detector(image_3)
#
# output_4, num_points_4 = HCD.harris_corner_detector(image_4)
#
# np.savetxt('output_1.txt', output_1)
# np.savetxt('output_2.txt', output_2)
# np.savetxt('output_3.txt', output_3)
# np.savetxt('output_4.txt', output_4)

# Match Features
output_1 = np.loadtxt('output_1.txt')
output_2 = np.loadtxt('output_2.txt')
output_3 = np.loadtxt('output_3.txt')
output_4 = np.loadtxt('output_4.txt')

HCD.plot_harris_points(output_1, image_2)
HCD.plot_harris_points(output_2, image_3)
HCD.plot_harris_points(output_2, image_1)
HCD.plot_harris_points(output_2, image_4)

matches = mf.match_features(image_1, output_1, image_2, output_2)

# Run ransac
H, robust_matches = RANSAC(output_1, output_2, matches)

# stitch images together
warped_image_1 = cv2.warpPerspective(image_1, H, ((image_1.shape[1] + image_2.shape[1]), image_2.shape[0]))
warped_image_1[0:image_2.shape[0], 0:image_2.shape[1]] = image_2

# repeat
output_5, num_points_5 = HCD.harris_corner_detector(warped_image_1)
matches_1 = mf.match_features(warped_image_1, output_5, image_3, output_3)
H, robust_matches = RANSAC(output_5, output_3, matches_1)

# stitch images together
warped_image_2 = cv2.warpPerspective(warped_image_1, H, ((warped_image_1.shape[1] + image_3.shape[1]), image_3.shape[0]))
warped_image_2[0:image_3.shape[0], 0:image_3.shape[1]] = image_3

# repeat last time
output_6, num_points_6 = HCD.harris_corner_detector(warped_image_2)
matches_2 = mf.match_features(warped_image_2, output_6, image_4, output_4)
H, robust_matches = RANSAC(output_6, output_4, matches_2)

# stitch images together
warped_image_final = cv2.warpPerspective(warped_image_2, H, ((warped_image_2.shape[1] + image_4.shape[1]), image_4.shape[0]))
warped_image_final[0:image_4.shape[0], 0:image_4.shape[1]] = image_4

plt.imshow(warped_image_final)
plt.show()



