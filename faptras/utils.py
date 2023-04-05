from typing import List, Tuple
import os
import time
import sys
import math
from enum import Enum
from functools import total_ordering
from collections import defaultdict

import numpy as np
import cv2 as cv
import pandas as pd

import globals


@total_ordering
class SprintCategory(Enum):
    def __le__(self, b):
        return self.value <= b.value

    WALKING = 1
    EASY = 2
    MODERATE = 3
    FAST = 4
    VERY_FAST = 5


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
    def scaler_homo_func(row): return [int(
        row[0] / row[2]), int(row[1] / row[2])]
    start = False
    with open(path_to_detections, "r") as d_file:
        lines = d_file.readlines()
        for line in lines:
            line = line.split(" ")
            # Extract bounding box information
            if not start:
                start = True
                start_frame_id = int(line[0])
            frame_id = int(line[0]) - start_frame_id + 1
            object_id = int(line[1])
            bb_left = float(line[2])
            bb_top = float(line[3])
            bb_width = float(line[4])
            bb_height = float(line[5])
            # Calculate lower center
            if frame_id > last_frame_id:
                detections = np.array(detections)@H.T
                detections = np.apply_along_axis(
                    scaler_homo_func, 1, detections)
                assert detections.shape[0] == len(objects) == len(bb_info)
                storage[last_frame_id] = (detections, bb_info, objects)
                detections, objects, bb_info = [], [], []
            # For every frame do
            detections.append(
                [bb_left + 0.5 * bb_width, bb_top + bb_height, 1])
            objects.append(object_id)
            bb_info.append([int(bb_left), int(bb_top),
                           int(bb_width), int(bb_height)])
            last_frame_id = frame_id
        print(
            f"Reading: {frame_id} frames took: {(time.time() - start_time):.2f}s")
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

def to_tuple_float(coords: Tuple) -> Tuple[float, float]:
    return float(coords[0]), float(coords[1])

def count_not_seen_players(match_, missing_ids: List[int], frame_id: int):
    unseen_frames = []
    for missing_id in missing_ids:
        unseen_frames.append(
            frame_id - match_.find_person_with_id(missing_id).last_seen_frame_id)
    return unseen_frames

def calculate_euclidean_distance(current_position: Tuple[float, float], new_position: Tuple[float, float]):
    return math.sqrt((current_position[0] - new_position[0])**2 + (current_position[1] - new_position[1])**2)

def convert_frame_to_minutes(frame, fps_rate):
    return frame / (fps_rate * 60.0)

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

def metabolic_cost(acc: np.array):
    """Calculates metabolic cost from provided accelerations.
    """ 
    cost = np.zeros_like(acc)
    for i in range(acc.size):
        if acc[i] > 0:
            cost[i] = 0.102 * ((acc[i] ** 2 + 96.2) ** 0.5) * (4.03 * acc[i] + 3.6 * np.exp(-0.408 * acc[i]))
        elif acc[i] < 0:
            cost[i] =  0.102 * ((acc[i] ** 2 + 96.2) ** 0.5) * (-0.85 * acc[i] + 3.6 * np.exp(1.33 * acc[i]))
        else:
            acc[i] = 0.0
    return cost


def extract_sequences(arr: np.array) -> pd.DataFrame:
    """Extract sequences from array and puts it in the DataFrame by saving the size of each sequence.

    Args:
        arr (np.array): Source array

    Returns:
        pd.DataFrame: Resulting DataFrame.
    """
    # [0] is neeeded because of the returned array format
    sequences = np.split(arr, np.where(np.diff(arr) != 0)[0] + 1)
    data = defaultdict(list)
    for seq in sequences:
        data[seq[0]].append(len(seq))
    return data
    
