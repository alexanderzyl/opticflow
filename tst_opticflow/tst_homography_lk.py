import cv2
import numpy as np
import pytest
import quaternion
import copy

from homography.decomposition import HomographyDecomposition
from homography.optical_flow import features_to_track, checked_trace, trace_homography


@pytest.fixture
def video_src():
    video_src = cv2.VideoCapture("../data/output-2.mp4")
    return video_src


def line3d(img, pt1, pt2, K):
    objpt = np.float64([pt1,pt2])
    imgpt0, _ = cv2.projectPoints(objpt, np.zeros(3), np.zeros(3), K, np.float64([]))
    p1 = tuple(imgpt0[0].ravel())
    p2 = tuple(imgpt0[1].ravel())
    img = cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255,50,100), 3)

green = (0, 255, 0)

# optical axis
x_c = 627.15663904
y_c = 479.18418123

dist = np.array([-0.05934245, -0.01461297, -0.03792086, 0.00428712, 0.00299862], dtype=float)

K = [[288.74930952, 0., x_c],
     [0., 297.66981327, y_c],
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

    #undistortion = False
    undistortion = True
    frame_num = 500

    video_src.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    _, frame0 = video_src.read()
    video_src.set(cv2.CAP_PROP_POS_FRAMES, frame_num + 10)
    _, frame1 = video_src.read()

    Cam = copy.copy(K)

    if (undistortion == True):
        h,  w = frame0.shape[:2]
        newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
        frame0 = cv2.undistort(frame0, K, dist, None, newK)
        frame1 = cv2.undistort(frame1, K, dist, None, newK)
        Cam = copy.copy(newK)

    features = features_to_track(frame0)
    assert features is not None

    features_update, keep = checked_trace(frame0, frame1, features)
    assert features_update is not None

    live_features = features[keep]
    live_features_update = features_update[keep]

    H, status = trace_homography(live_features, live_features_update, True)
    assert H is not None

    # Decompose
    HD = HomographyDecomposition(H, Cam)

    diffs = [np.linalg.norm(H - Hr) for Hr in HD.H_r]

    best = 2

    H_best = HD.H_r[best]
    # H_best = HD.H_r[0]

    # Convert the rotation matrix R to a rotation vector
    rvec, _ = cv2.Rodrigues(HD.rotations[best])

    # Sample translation vector (replace with your actual translation vector)
    tvec = np.array([0, 0, 0], dtype=float)

    # Distortion coefficients (replace with your actual distortion coefficients or use zeros if unknown)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=float)

    if (undistortion == True):
       dist_coeffs = dist

    h, w = frame1.shape[:2]
    overlay = cv2.warpPerspective(frame0, H_best, (w, h))
    # overlay = frame0.copy()
    vis = cv2.addWeighted(frame1, 0.5, overlay, 0.5, 0.0)

    # Draw the axes on the image
    vis = cv2.drawFrameAxes(
        vis, Cam, dist_coeffs, np.zeros(3), np.zeros(3), length=1.0, thickness=1
    )

    translation_res = HD.translations[best]
    normal_res = HD.normals[best]

    print("normals:")
    print(HD.normals)

    print("best normal:")
    print(HD.normals[best])

    x = translation_res[0][0]
    y = translation_res[1][0]
    z = translation_res[2][0]

    line3d(img=vis, pt1=[0.,0.,0.], pt2=[x,y,z], K=Cam)

    cv2.imshow('decomposed_H', vis)
    cv2.waitKey(0)


def test_2_successive_rotations(video_src):
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 550)
    _, frame0 = video_src.read()
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 560)
    _, frame1 = video_src.read()
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 570)
    _, frame2 = video_src.read()

    features0 = features_to_track(frame0)

    features_update, keep = checked_trace(frame0, frame1, features0)
    features1 = features0[keep]
    live_features_update = features_update[keep]
    H_01, status = trace_homography(features1, live_features_update, True)
    HD_01 = HomographyDecomposition(H_01, K)
    Q_01 = [quaternion.from_rotation_matrix(R) for R in HD_01.rotations]

    features_update, keep = checked_trace(frame1, frame2, features1)
    features2 = features1[keep]
    live_features_update = features_update[keep]
    H_12, status = trace_homography(features2, live_features_update, True)
    HD_12 = HomographyDecomposition(H_12, K)
    Q_12 = [quaternion.from_rotation_matrix(R) for R in HD_12.rotations]

    # cartesian product of Q_01 and Q_12
    Q_012 = [q2 * q1 for q1 in Q_01 for q2 in Q_12]

    features_update, keep = checked_trace(frame0, frame2, features0)
    features2 = features0[keep]
    live_features_update = features_update[keep]
    H_02, status = trace_homography(features2, live_features_update, True)
    HD_02 = HomographyDecomposition(H_02, K)
    Q_02 = [quaternion.from_rotation_matrix(R) for R in HD_02.rotations]

    # For each pair in Q_012 and Q_02 find the quaternion distance. Return the pair with the smallest distance and
    # the index in Q_02
    t = [(1 - (q.components @ Q_02[i].components) ** 2, q, i) for q in Q_012 for i in range(len(Q_02))]
    t.sort(key=lambda x: x[0])
    q012 = t[0][1]
    hd02_index = t[0][2]

    H_best = HD_02.reconstruct(quaternion.as_rotation_matrix(q012),
                               HD_02.translations[hd02_index], HD_02.normals[hd02_index])

    # diffs = [np.linalg.norm(H_02 - Hr) for Hr in HD_02.H_r]
    #
    # best = np.argmin(diffs)
    #
    # H_best = HD_02.H_r[best]

    h, w = frame2.shape[:2]
    overlay = cv2.warpPerspective(frame0, H_best, (w, h))
    vis = cv2.addWeighted(frame2, 0.5, overlay, 0.5, 0.0)

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
