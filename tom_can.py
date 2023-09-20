# example of Tomasi-Kanade factorization
# 2D points to 3D points
# usage: python tom_can.py <2D points file> <3D points file>
import numpy as np
import cv2 as cv
import sys
import math
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.cluster import KMeans

cam = cv.VideoCapture("output-1.mp4")
ret, prev = cam.read()
prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

coords = np.array([
    [230, 218, 205, 189, 176, 156],
    [145, 156, 162, 166, 166, 165]
])

total_magnitude = []

while True:
    ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev=prevgray,
                                        next=gray,
                                        flow=None,
                                        pyr_scale=0.5,
                                        levels=10,
                                        winsize=15,
                                        iterations=3,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    print (mag)
    print (ang)
    print ("frame")

    prevgray = gray

    if cv.waitKey(10) == 27:
        break
cv.destroyAllWindows()
cam.release()
#
#             vis = frame.copy()
#
#             if len(self.tracks) > 0:
#                 img0, img1 = self.prev_gray, frame_gray
#                 p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
#                 p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
#                 p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
#                 d = abs(p0-p0r).reshape(-1, 2).max(-1)
#                 good = d < 1
#                 new_tracks = []

#                 for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
#                     if not good_flag:
#                         continue
#                     tr.append((x, y))
#                     if len(tr) > self.track_len:
#                         del tr[0]
#                     new_tracks.append(tr)
#                     cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
#                 self.tracks = new_tracks
#                 cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
#                 draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

#             if self.frame_idx % self.detect_interval == 0:
#                 mask = np.zeros_like(frame_gray)
#                 mask[:] = 255
#                 for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
#                     cv.circle(mask, (x, y), 5, 0, -1)
#                 p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
#                 if p is not None:
#                     for x, y in np.float32(p).reshape(-1, 2):
#                         self.tracks.append([(x, y)])
#
#             self.frame_idx += 1
#             self.prev_gray = frame_gray

#             cv.imshow('lk_track', vis)
#
#             ch = cv.waitKey(1)
#             if ch == 27:
#                 break

# def main():
#     import sys
#     try:
#         video_src = sys.argv[1]
#     except:
#         video_src = 0

#     App(video_src).run()
#     print('Done')


# if __name__ == '__main__':
#     print(__doc__)
#     main()
#     cv.destroyAllWindows()

