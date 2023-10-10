import cv2
import numpy as np
import pytest

from homography.optical_flow import features_to_track, checked_trace, trace_homography


@pytest.fixture
def video_src():
    video_src = cv2.VideoCapture("../data/output-2.mp4")
    return video_src


green = (0, 255, 0)

K = [[288.74930952, 0., 627.15663904],
     [0., 297.66981327, 479.18418123],
     [0., 0., 1.]]

K = np.array(K)


def test_2_frames(video_src):
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 600)
    _, frame0 = video_src.read()
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 610)
    _, frame1 = video_src.read()

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

    for (x0, y0), (x1, y1) in zip(live_features[:, 0], live_features_update[:, 0]):
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
        cv2.circle(vis, (x1, y1), 2, green, -1)

    cv2.imshow('lk_homography', vis)
    cv2.waitKey(0)


def test_decompose_H(video_src):
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 600)
    _, frame0 = video_src.read()
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 610)
    _, frame1 = video_src.read()

    features = features_to_track(frame0)
    assert features is not None

    features_update, keep = checked_trace(frame0, frame1, features)
    assert features_update is not None

    live_features = features[keep]
    live_features_update = features_update[keep]

    H, status = trace_homography(live_features, live_features_update, True)
    assert H is not None

    # Decompose
    retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    def _reconstruct_H(solution):
        R = rotations[solution]
        t = translations[solution]
        n = normals[solution]

        # Reconstruct H
        H_r = K @ (R - t @ n.T) @ np.linalg.inv(K)

        # find the distance between H and H_r
        return H_r

    diffs = [np.linalg.norm(H - _reconstruct_H(solution)) for solution in range(len(rotations))]

    H_r = _reconstruct_H(np.argmin(diffs))
    # H_r = _reconstruct_H(3)

    h, w = frame1.shape[:2]
    overlay = cv2.warpPerspective(frame0, H_r, (w, h))
    # overlay = frame0.copy()
    vis = cv2.addWeighted(frame1, 0.5, overlay, 0.5, 0.0)

    cv2.imshow('decomposed_H', vis)
    cv2.waitKey(0)


def test_full_run(video_src):
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 0)
    res, frame0 = video_src.read()
    res, frame1 = video_src.read()
    i = 0
    while res:
        if i % 10 == 0:
            features = features_to_track(frame0)
            features_update, keep = checked_trace(frame0, frame1, features)
            live_features = features[keep]
            live_features_update = features_update[keep]
            H, status = trace_homography(live_features, live_features_update, True)
        frame0 = frame1
        res, frame1 = video_src.read()
        i += 1

    print(i)
