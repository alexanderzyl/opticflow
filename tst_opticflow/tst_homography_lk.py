import cv2
import pytest


@pytest.fixture
def video_src():
    video_src = cv2.VideoCapture("../data/output-2.mp4")
    return video_src


def test_temp(video_src):
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 10)
    _, frame0 = video_src.read()
    video_src.set(cv2.CAP_PROP_POS_FRAMES, 11)
    _, frame1 = video_src.read()

    cv2.imshow('frame0', frame0)
    cv2.imshow('frame1', frame1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
