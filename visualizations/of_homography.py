'''
Lucas-Kanade homography tracker
===============================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames. Finds homography between reference and current views.

Usage
-----
lk_homography.py [<video_source>]


Keys
----
ESC   - exit
SPACE - start tracking
r     - toggle RANSAC
'''

# https://github.com/npinto/opencv/blob/master/samples/python2/lk_homography.py

import cv2

from homography.optical_flow import checked_trace, trace_homography

feature_params = dict(maxCorners=1000,
                      qualityLevel=0.01,
                      minDistance=8,
                      blockSize=19)

green = (0, 255, 0)
red = (0, 0, 255)


class App:
    def __init__(self, video_src):
        self.cam = video_src
        self.start_features = None
        self.cur_features = None
        self.use_ransac = True

    def run(self):
        while True:
            _, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            if self.start_features is not None:
                p2, trace_status = checked_trace(self.gray1, frame_gray, self.cur_features)

                self.cur_features = p2[trace_status].copy()
                self.start_features = self.start_features[trace_status].copy()
                self.gray1 = frame_gray

                if len(self.start_features) < 4:
                    self.start_features = None
                    continue
                H, status = trace_homography(self.start_features, self.cur_features, self.use_ransac)
                h, w = frame.shape[:2]
                overlay = cv2.warpPerspective(self.frame0, H, (w, h))
                vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0.0)

                self.vis_vectors(status, vis)
            else:
                self.vis_features(frame_gray, vis)

            cv2.imshow('lk_homography', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            if ch == ord(' '):
                self.init_features(frame, frame_gray)
            if ch == ord('r'):
                self.use_ransac = not self.use_ransac

    def vis_vectors(self, status, vis):
        for (x0, y0), (x1, y1), good in zip(self.start_features[:, 0], self.cur_features[:, 0], status[:, 0]):
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
            if good:
                cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
            cv2.circle(vis, (x1, y1), 2, (red, green)[good], -1)
        # draw_str(vis, (20, 20), 'track count: %d' % len(self.p1))
        # if self.use_ransac:
        #     draw_str(vis, (20, 40), 'RANSAC')

    def vis_features(self, frame_gray, vis):
        p = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if p is not None:
            for x, y in p[:, 0]:
                # cv2.circle(vis, (x, y), 2, green, -1)
                cv2.circle(vis, (int(x), int(y)), 2, green, 2)
            # draw_str(vis, (20, 20), 'feature count: %d' % len(p))

    def init_features(self, frame, frame_gray):
        self.frame0 = frame.copy()
        self.start_features = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if self.start_features is not None:
            self.cur_features = self.start_features
            self.gray0 = frame_gray
            self.gray1 = frame_gray


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = cv2.VideoCapture("../data/output-2.mp4")

    # print __doc__
    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
