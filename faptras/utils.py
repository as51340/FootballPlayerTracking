import os
import time
import sys

import numpy as np
import cv2 as cv

import globals

def get_file_name(path: str) -> str:
    """Extracts file name from the path. E.g. for path /user/video/t7.mp4 returns t7

    Args:
        path (str): path to the file

    Returns:
        str: Extracted file name.
    """
    last_slash = path.rfind('/')
    last_dot = path.rfind('.')
    return path[last_slash + 1:last_dot]

def squash_detections(path_to_detections: str, H: np.ndarray):
    """Squashes detections from the bounding box to the one point and transforms them using the homography matrix.
    Args:
        path_to_detections (str): 
        H (np.ndarray): Homography matrix.
    """
    start_time = time.time()
    storage = dict()
    last_frame_id = sys.maxsize
    detections, objects, bb_info = [], [], []
    scaler_homo_func = lambda row: [int(row[0] / row[2]), int(row[1] / row[2])]
    with open(path_to_detections, "r") as d_file:
        lines = d_file.readlines()
        for line in lines:
            line = line.split(" ")
            # Extract bounding box information
            frame_id = int(line[0])
            object_id = int(line[1])
            bb_left = float(line[2])
            bb_top = float(line[3])
            bb_width = float(line[4])
            bb_height = float(line[5])
            # Calculate lower center
            if frame_id > last_frame_id:
                detections = np.array(detections)@H.T
                detections = np.apply_along_axis(scaler_homo_func, 1, detections)
                assert detections.shape[0] == len(objects) == len(bb_info)
                storage[last_frame_id] = (detections, bb_info, objects)
                detections, objects, bb_info = [], [], []
            # For every frame do
            detections.append([bb_left + 0.5 * bb_width, bb_top + bb_height, 1])
            objects.append(object_id)
            bb_info.append([int(bb_left), int(bb_top), int(bb_width), int(bb_height)])
            last_frame_id = frame_id
        print(f"Reading: {frame_id} frames took: {time.time() - start_time}s")
        return storage

def check_kill(k):
    if k == ord('k'):
        os._exit(-1)
        
def pause():
    while True:
        time.sleep(0.5)
        k = cv.waitKey(1) & 0xFF
        check_kill(k)
        if k == ord('p') or globals.stop_thread:
            break
