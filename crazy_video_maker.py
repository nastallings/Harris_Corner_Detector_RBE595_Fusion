import cv2
import Harris_Corner_Detector as hcd
import matplotlib.pyplot as plt

vid = cv2.VideoCapture('crazy_corner_video.mp4')
result = cv2.VideoWriter("crazy_video_output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 10, (int(vid.get(3)), int(vid.get(4))))

while vid.isOpened():
    try:
        ret, frame = vid.read()
        points, count = hcd.harris_corner_detector(frame)
        for element in points:
            frame[int(element[0])][int(element[1])] = [int(element[0]), int(element[1]), 0]
        drawable_image = cv2.putText(frame, "Number of Corners: " + str(count), (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        result.write(cv2.cvtColor(drawable_image, cv2.COLOR_BGR2RGB))
    except ValueError:
        print("exception")

result.release()

