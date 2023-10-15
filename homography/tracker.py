from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import quaternion

from homography.decomposition import HomographyDecomposition
from homography.optical_flow import features_to_track, checked_trace, trace_homography


class Tracker:
    def __init__(self):
        self._prev_frame = None
        self._cur_frame = None
        self._features_to_track = None
        self._H_status = None
        self._H = None
        self._quaternion = quaternion.one
        self._decomposition = None

    @property
    def prev_frame(self):
        return self._prev_frame

    @prev_frame.setter
    def prev_frame(self, frame):
        self._prev_frame = frame
        if self._prev_frame is not None:
            self._features_to_track = features_to_track(self._prev_frame)

    @property
    def cur_frame(self):
        return self._cur_frame

    @cur_frame.setter
    def cur_frame(self, frame):
        self.prev_frame = self._cur_frame
        self._cur_frame = frame

    @property
    def decomposition(self):
        return self._decomposition

    @property
    def accum_quaternion(self):
        return self._quaternion

    def update(self, K):
        if self.prev_frame is not None and self.cur_frame is not None and self._features_to_track is not None:
            features_update, keep = checked_trace(self.prev_frame, self.cur_frame, self._features_to_track)
            live_features = self._features_to_track[keep]
            live_features_update = features_update[keep]
            self._H, self._H_status = trace_homography(live_features, live_features_update, True)
            self._decompose(K)

    def _decompose(self, K):
        # Decompose
        if self._H is None:
            return None

        HD = HomographyDecomposition(self._H, K)

        diffs = [np.linalg.norm(self._H - Hr) for Hr in HD.H_r]

        best = np.argmin(diffs)

        cur_quaternion = quaternion.from_rotation_matrix(HD.rotations[best])
        self._quaternion = cur_quaternion * self._quaternion

        self._decomposition = SimpleNamespace(**{
            'R': HD.rotations[best],
            'Ho': HD.H,
            'Hr': HD.H_r[best],
            't': HD.translations[best],
            'n': HD.normals[best]})