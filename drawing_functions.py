import numpy as np
import cv2

"""
===========================================================================================
=========================   Functions and general description    ==========================
===========================================================================================

mark_points(n: int, img: np.ndarray):  marks n points on image (or less if 'q' was pressed)
"""


def mark_points(n: int, img: np.ndarray):
    """
    marks n points on image (or less if 'q' was pressed)
    :param img: image to mark pts on
    :param n: bound num
    """
    def get_points(event, x, y, flags, params):
        global arr, check
        if event == cv2.EVENT_LBUTTONDOWN:
            arr.append([x, y])
            check += 1
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            cv2.imshow("image", img)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_points)
    check = 0
    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif check == n:
            break