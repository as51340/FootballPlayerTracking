from typing import Dict, List, Tuple
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

def to_tuple_int(coords: Tuple) -> Tuple[int, int]:
    return int(coords[0]), int(coords[1])

def get_existing_objects(detections_in_pitch: List[Tuple[int, int]], bb_info_in_pitch: List[Tuple[int, int, int, int]], object_ids_in_pitch: List[int], new_objects_id: List[int]):
    """This method returns information about all objects that don't need to be resolved and about which we already have some information.

    Args:
        detections_in_pitch (List[Tuple[int, int]]): De
        bb_info_in_pitch (List[Tuple[int, int, int, int]]): _description_
        new_object_id (List[int]): _description_
        object_ids_in_pitch (List[int]): _description_

    Returns:
        _type_: _description_
    """
    existing_objects_detections, existing_objects_bb_info, existing_objects_ids = [], [], []
    for i in range(len(object_ids_in_pitch)):
        if object_ids_in_pitch[i] not in new_objects_id:
            existing_objects_detections.append(detections_in_pitch[i])
            existing_objects_bb_info.append(bb_info_in_pitch[i])
            existing_objects_ids.append(object_ids_in_pitch[i])
    return existing_objects_detections, existing_objects_bb_info, existing_objects_ids
    