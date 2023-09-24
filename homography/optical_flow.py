import cv2


def checked_trace(img_ref, img_new, features, back_threshold=1.0):
    lk_params = dict(winSize=(19, 19),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    features_update, _st, _err = cv2.calcOpticalFlowPyrLK(img_ref, img_new, features, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img_new, img_ref, features_update, None, **lk_params)
    d = abs(features - p0r).reshape(-1, 2).max(-1)
    keep = d < back_threshold
    return features_update, keep


def trace_homography(features0, features1, use_ransac):
    H, status = cv2.findHomography(features0, features1, cv2.RANSAC if use_ransac else 0, 10.0)
    return H, status


def features_to_track(frame):
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.01,
                          minDistance=8,
                          blockSize=19)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.goodFeaturesToTrack(frame_gray, **feature_params)


