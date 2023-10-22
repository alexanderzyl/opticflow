import copy

import cv2
import numpy as np
import pytest
from skimage.transform import radon, iradon

from homography.decomposition import HomographyDecomposition
from homography.optical_flow import features_to_track, checked_trace, trace_homography


@pytest.fixture
def video_src():
    video_src = cv2.VideoCapture("../data/output-2.mp4")
    return video_src


@pytest.fixture
def K():
    # optical axis
    x_c = 627.15663904
    y_c = 479.18418123

    ret = [[288.74930952, 0., x_c],
           [0., 297.66981327, y_c],
           [0., 0., 1.]]

    ret = np.array(ret)
    return ret


@pytest.fixture
def dist():
    return np.array([-0.05934245, -0.01461297, -0.03792086, 0.00428712, 0.00299862], dtype=float)


green = (0, 255, 0)


def line3d(img, pt1, pt2, K):
    objpt = np.float64([pt1, pt2])
    imgpt0, _ = cv2.projectPoints(objpt, np.zeros(3), np.zeros(3), K, np.float64([]))
    p1 = tuple(imgpt0[0].ravel())
    p2 = tuple(imgpt0[1].ravel())
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 50, 100), 3)


def test_sinogram_2_frames(video_src):
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 500)
    _, frame0 = video_src.read()
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 510)
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

    v0, v1 = live_features[:, 0], live_features_update[:, 0]
    n_v = len(v0)
    n_v = np.sqrt(n_v).astype(int)
    vectors = np.zeros((h, w), dtype=np.uint8)

    for (x0, y0), (x1, y1) in zip(v0, v1):
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        cv2.line(vectors, (x0, y0), (x1, y1), 255, 1)
        cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
        cv2.circle(vis, (x1, y1), 2, green, -1)

    sinogram = radon(vectors, theta=np.arange(0, 180, 5), circle=True)
    # Find the threshold value directly on the multi-dimensional array
    threshold = np.partition(sinogram.ravel(), -n_v)[-n_v]

    # Set values below the threshold to zero
    sinogram[sinogram < threshold] = 0
    vectors = iradon(sinogram, theta=np.arange(0, 180, 5), circle=True)

    # find the pixel with the largest value
    max_value = vectors.max()
    y, x = np.where(vectors == max_value)
    cv2.circle(vis, (x[0], y[0]), 2, (0, 0, 255), -1)

    from matplotlib import pyplot as plt
    plt.imshow(vectors, cmap='gray')
    plt.show()

    cv2.imshow('lk_homography', vis)
    cv2.waitKey(0)


def test_full_run(video_src, K, dist):
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 0)
    res, frame0 = video_src.read()
    res, frame1 = video_src.read()
    i = 0

    h, w = frame0.shape[:2]
    new_cam, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    freeze = False

    while res:
        if i % 10 == 0:
            frame0 = cv2.undistort(frame0, K, dist, None, new_cam)
            frame1 = cv2.undistort(frame1, K, dist, None, new_cam)
            features = features_to_track(frame0)
            assert features is not None

            features_update, keep = checked_trace(frame0, frame1, features)
            assert features_update is not None

            live_features = features[keep]
            live_features_update = features_update[keep]

            H, status = trace_homography(live_features, live_features_update, True)
            assert H is not None

            # Decompose
            HD = HomographyDecomposition(H, new_cam)

            best = 2
            H_best = HD.H_r[best]

            h, w = frame1.shape[:2]
            overlay = cv2.warpPerspective(frame0, H_best, (w, h))

            vis = cv2.addWeighted(frame1, 0.5, overlay, 0.5, 0.0)

            # Draw the axes on the image
            vis = cv2.drawFrameAxes(
                vis, new_cam, dist, np.zeros(3), np.zeros(3), length=1.0, thickness=1
            )

            translation_res = HD.translations[best]

            x = translation_res[0][0]
            y = translation_res[1][0]
            z = translation_res[2][0]

            line3d(img=vis, pt1=[0., 0., 0.], pt2=[x, y, z], K=new_cam)

            v0, v1 = live_features[:, 0], live_features_update[:, 0]

            for (x0, y0), (x1, y1) in zip(v0, v1):
                x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
                cv2.line(vis, (x0, y0), (x1, y1), (200, 0, 0), 2)
                cv2.circle(vis, (x1, y1), 2, green, -1)

            cv2.imshow('decomposed_H', vis)
            key = cv2.waitKey(0) if freeze else cv2.waitKey(100)
            if key == 27:
                break
            if key == ord(' '):
                freeze = not freeze
            cv2.destroyAllWindows()

        frame0 = frame1
        res, frame1 = video_src.read()
        i += 1

    print(i)
