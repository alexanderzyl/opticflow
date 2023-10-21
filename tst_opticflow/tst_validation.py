import cv2
import numpy as np
import pytest
from skimage.transform import radon, iradon

from homography.optical_flow import features_to_track, checked_trace, trace_homography


@pytest.fixture
def video_src():
    video_src = cv2.VideoCapture("../data/output-2.mp4")
    return video_src


green = (0, 255, 0)


def test_2_frames(video_src):
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 600)
    _, frame0 = video_src.read()
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 610)
    _, frame1 = video_src.read()

    # find the minimum of w and h, devide by 2, and resize the image to that size
    minv = min(frame0.shape[0], frame0.shape[1])
    frame0 = cv2.resize(frame0, (minv // 2, minv // 2))
    frame1 = cv2.resize(frame1, (minv // 2, minv // 2))

    features = features_to_track(frame0)
    assert features is not None

    features_update, keep = checked_trace(frame0, frame1, features)
    assert features_update is not None

    live_features = features[keep]
    live_features_update = features_update[keep]

    H, status = trace_homography(live_features, live_features_update, True)
    # live_features = live_features[:, 0, :][status]
    # live_features_update = live_features_update[:, 0, :][status]
    assert H is not None

    h, w = frame1.shape[:2]
    overlay = cv2.warpPerspective(frame0, H, (w, h))
    vis = cv2.addWeighted(frame1, 0.5, overlay, 0.5, 0.0)

    # for (x0, y0), (x1, y1) in zip(live_features[:, 0], live_features_update[:, 0]):
    #     x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
    #     cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
    #     cv2.circle(vis, (x1, y1), 2, green, -1)

    v0, v1 = live_features[:, 0], live_features_update[:, 0]
    v0 = v0.astype(np.int32)
    v1 = v1.astype(np.int32)

    vectors = np.zeros((h, w), dtype=np.uint8)
    lines = cv2.HoughLines(vectors, 1, np.pi / 180, 128)
    # Draw the detected lines
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(vectors, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Draw detected lines in blue


    # draw lines on the image using OpenCV v0 - first points, v1 - second points
    # Draw lines
    for (x1, y1), (x2, y2) in zip(v0, v1):
        cv2.line(vectors, (x1, y1), (x2, y2), 255, 1)


    from matplotlib import pyplot as plt
    plt.imshow(vectors, cmap='gray')
    plt.show()

    cv2.imshow('lk_homography', vis)
    cv2.waitKey(0)

