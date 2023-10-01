import numpy as np
import cv2
import glob
import os

green = (0, 255, 0)
red = (0, 0, 255)

class App:
    def __init__(self, video_src):
        self.cam = cv2.VideoCapture("/Users/Vadim/Desktop/CV/pkk/DJIG0006.mp4")
        self.p0 = None

    def video2Images(self):
        ind = 0
        while True:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            filename = "/Users/Vadim/Desktop/CV/pkk/" + str(ind) + ".png"
            cv2.imwrite(filename, frame_gray)
            ind += 1

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def calibrate():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((12*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:12].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = load_images_from_folder("./calib/")

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11,12), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,12), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (11,12), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('Calibration done')

    #Undistortion example
    # img = cv2.imread('/Users/Vadim/Desktop/CV/pkk/calib/750.png')
    # h,  w = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # # crop the image
    # #x, y, w, h = roi
    # #dst = dst[y:y+h, x:x+w]
    # cv2.imwrite('/Users/Vadim/Desktop/CV/pkk/750.png', dst)
    # print('Undistortion done')

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    calibrate()
    #App(video_src).video2Images()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
