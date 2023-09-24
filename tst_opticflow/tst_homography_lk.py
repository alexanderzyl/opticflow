import cv2
import pytest

from homography.optical_flow import features_to_track, checked_trace, trace_homography


@pytest.fixture
def video_src():
    video_src = cv2.VideoCapture("../data/output-2.mp4")
    return video_src


def test_2_frames(video_src):
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 10)
    _, frame0 = video_src.read()
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 11)
    _, frame1 = video_src.read()
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    features = features_to_track(frame0)
    assert features is not None

    features_update, keep = checked_trace(frame0, frame1, features)
    assert features_update is not None

    live_features = features[keep]
    live_features_update = features_update[keep]

    H, status = trace_homography(live_features, live_features_update, True)
    assert H is not None