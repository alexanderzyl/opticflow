import cv2
import numpy as np


class HomographyDecomposition:
    def __init__(self, H, K):
        self.H = H
        self.K = K
        self.status, self.rotations, self.translations, self.normals = cv2.decomposeHomographyMat(H, K)
        self.H_r = [self.reconstruct(R, t, n) for R, t, n in zip(self.rotations, self.translations, self.normals)]

    def reconstruct(self, R, t, n):
        H_r = self.K @ (R - t @ n.T) @ np.linalg.inv(self.K)
        return H_r

    def __len__(self):
        return len(self.rotations)
