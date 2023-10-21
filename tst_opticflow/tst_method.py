import itertools

import cv2
import numpy as np
import pytest
import quaternion

from homography.tracker import Tracker

K = [[288.74930952, 0., 627.15663904],
     [0., 297.66981327, 479.18418123],
     [0., 0., 1.]]

K = np.array(K)


def frame_iterator(video_src):
    while True:
        ret, frame = video_src.read()
        if not ret:
            break
        yield frame


@pytest.fixture
def video_src():
    video_src = cv2.VideoCapture("../data/output-2.mp4")
    return frame_iterator(video_src)


def draw_axes(R, vis):
    # Draw the axes on the image
    # Convert the rotation matrix R to a rotation vector
    rvec, _ = cv2.Rodrigues(R)
    # Sample translation vector (replace with your actual translation vector)
    tvec = np.array([0, 0, 0], dtype=float)
    # Distortion coefficients (replace with your actual distortion coefficients or use zeros if unknown)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=float)
    vis = cv2.drawFrameAxes(
        vis, K, dist_coeffs, rvec, tvec, length=1.0, thickness=3
    )
    return vis


def test_visual(video_src):
    method = Tracker()

    for frame in itertools.islice(video_src, 0, None, 10):
        method.cur_frame = frame
        method.update(K)
        if method.decomposition is not None:
            decomposed = method.decomposition
            h, w = method.cur_frame.shape[:2]
            overlay = cv2.warpPerspective(method.prev_frame, decomposed.Hr, (w, h))

            vis = cv2.addWeighted(method.cur_frame, 0.5, overlay, 0.5, 0.0)
            R = quaternion.as_rotation_matrix(method.accum_quaternion)
            # print(np.linalg.norm(method.accum_quaternion.components))
            # R =decomposed.R
            # vis = draw_axes(R, vis)
            cv2.imshow("Frame", vis)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break