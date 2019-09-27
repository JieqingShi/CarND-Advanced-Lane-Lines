""" Module for camera calibration """

import os
import cv2
import numpy as np
import pickle
import glob
import warnings
import logging

logger = logging.getLogger("CalibCamera")
logger.setLevel(logging.INFO)

# Console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.setLevel("INFO")


# nx = 9
# ny = 6
pickle_file = "camera_cal/dist_pickle.p"

def find_obj_and_img_points(images, nx, ny, draw=False):
    """ Function to find object points and image points """
    # See https://github.com/udacity/CarND-Camera-Calibration
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[:nx, :ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if ret == True:
            logger.debug(f"Found object points and image points on image {fname}")
            objpoints.append(objp)
            imgpoints.append(corners)
            # Not necessary to draw and display the corners
            if draw:
                cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
    
    logger.info(f"Found corners on {len(objpoints)} images")
    if not objpoints or not imgpoints:  # aka empty lists
        logger.warn("No objpoints or imgpoints found")
    
    return objpoints, imgpoints


def calibrate_camera(objpoints, imgpoints, img_size):
    """ Calibrate camera given objpoints and imgpoints """
    # Since ret, rvecs and tvecs are not used elsewhere there is no need to return them
    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    logger.info(f"Returned mtx and dist from camera calibration")
#     if pickle:
#         dict_pickle = dict(objpoints=objpoints, imgpoints=imgpoints, mtx=mtx, dist=dist)
#         pickle.dump(dict_pickle, open(pickle_file, "wb"))  # path is hardcoded
#         logger.info(f"Persisted objpoints, imgpoints, mtx and dist in pickle file")
    return mtx, dist

    
