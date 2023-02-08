from typing import List, Tuple, Dict
import os
from collections import defaultdict
import time
import sys
import argparse

import numpy as np
import cv2 as cv

from monitor_utils import get_offset_to_second_monitor

# Constants
RED = (0, 0, 255)
# SRC_WINDOW and DST_WINDOW are used for homography calculation
# DETECTIONS_WINDOW is used for birds-eye view on the pitch
# VIDEO_WINDOW is used for playing video with detections
SRC_WINDOW, DST_WINDOW, DETECTIONS_WINDOW = "src", "dst", "det"
VIDEO_WINDOW = "vid"

# Global variables
last_clicked_windows = []


# mouse callback function
def _select_points_wrapper(event, x, y, _, params):
    """Wrapper for mouse callback.
    """
    global last_clicked_window
    window, points, img_copy = params
    if event == cv.EVENT_LBUTTONDOWN:
        last_clicked_windows.append(window)
        cv.circle(img_copy, (x,y), 5, RED, -1)
        points.append([x, y])

def read_image(img_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads image from the given path and creates its copy.
    Args:
        img_path (str): Path of the image.
    """
    # Registers a mouse callback function on a image's copy.
    img = cv.imread(img_path, -1)
    img_copy = img.copy()
    return img, img_copy

def get_plan_view(src_img: np.ndarray, dst_img: np.ndarray, src_points: List[List[int]], dst_points: List[List[int]]):
    """Warps original image.
    Args:
        src_points (List[List[int]]): Source points' storage.
        dst_points (List[List[int]]): Destination points' storage.
        src_img (numpy.ndarray): Source image representation.
        dst_img (numpy.ndarray): Destination image representation.
    Returns:
        warped image, homography matrix and the mask.
    """
    src_pts = np.array(src_points).reshape(-1,1,2)
    dst_pts = np.array(dst_points).reshape(-1,1,2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    H = np.array(H)
    plan_view = cv.warpPerspective(src_img, H, (dst_img.shape[1], dst_img.shape[0]))
    return plan_view, H, mask

def merge_views(src_img: np.ndarray, dst_img: np.ndarray, src_points: List[List[int]], dst_points: List[List[int]]):
    """Merges two views.
    Args:
        src_points (List[List[int]]): Source points' storage.
        dst_points (List[List[int]]): Destination points' storage.
        src_img (np.ndarray): Source image.
        dst_img (np.ndarray): Destination image.
    """
    plan_view, H, mask = get_plan_view(src_img, dst_img, src_points, dst_points)
    for i in range(0, dst_img.shape[0]):
        for j in range(0, dst_img.shape[1]):
            if plan_view[i, j, 0] == 0 and plan_view[i, j, 1] == 0 and plan_view[i, j, 2] == 0:
                plan_view[i, j, 0] = dst_img[i, j, 0]
                plan_view[i, j, 1] = dst_img[i, j, 1]
                plan_view[i, j, 2] = dst_img[i, j, 2]
    return plan_view, H, mask

def full_screen_on_monitor(window_name):
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    monitor_info = get_offset_to_second_monitor()
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.moveWindow(window_name, monitor_info[0], monitor_info[1]);

    
def visualize(src_points: List[List[int]], dst_points: List[List[int]], src_img: np.ndarray, src_img_copy: np.ndarray, dst_img: np.ndarray, dst_img_copy: np.ndarray):
    """Runs visualization in while loop.
    Args:
        src_points (List[List[int]]): Source points' storage.
        dst_points (List[List[int]]): Destination points' storage.
        src_img (numpy.ndarray): Source image representation.
        src_img_copy (numpy.ndarray): Copy of the source image representation.
        dst_img (numpy.ndarray): Destination image representation.
        dst_img_copy (numpy.ndarray): Copy of the destination image representation.
    """
    # Source window setup
    full_screen_on_monitor(SRC_WINDOW)
    cv.setMouseCallback(SRC_WINDOW, _select_points_wrapper, (SRC_WINDOW, src_points, src_img_copy))
    # Destination window setup
    cv.namedWindow(DST_WINDOW)
    cv.setMouseCallback(DST_WINDOW, _select_points_wrapper, (DST_WINDOW, dst_points, dst_img_copy))
    while(1):
        # Keep showing copy of the image because of the circles drawn
        cv.imshow(SRC_WINDOW, src_img_copy)
        cv.imshow(DST_WINDOW, dst_img_copy)
        k = cv.waitKey(1) & 0xFF
        if k == ord('h'):
            print('create plan view')
            plan_view, H, mask = get_plan_view(src_img, dst_img, src_points, dst_points)
            cv.imshow("plan view", plan_view) 
        elif k == ord('m'):
            print('merge views')
            merge = merge_views(src_img, dst_img, src_points, dst_points)      
            cv.imshow("merge", merge)        
        elif k == ord('d') and last_clicked_windows:
            undo_window = last_clicked_windows.pop()
            if undo_window == SRC_WINDOW and src_points:
                src_points.pop()
                # Reinitialize image
                src_img_copy = src_img.copy()
                cv.setMouseCallback(SRC_WINDOW, _select_points_wrapper, (SRC_WINDOW, src_points, src_img_copy))
                draw_old_circles(src_img_copy, src_points)
            elif undo_window == DST_WINDOW and dst_points:
                dst_points.pop()
                # Reinitialize image
                dst_img_copy = dst_img.copy()
                cv.setMouseCallback(DST_WINDOW, _select_points_wrapper, (DST_WINDOW, dst_points, dst_img_copy))
                draw_old_circles(dst_img_copy, dst_points)
        elif k == ord("s"):
            print(f"Source points: {src_points}")
            print(f"Dest points: {dst_points}")
        elif k == ord('q'):
            cv.destroyAllWindows()
            return H

def draw_old_circles(img, points):
    """Draws all saved points on an image. Used for implementing undo buffer.
    Args:
        img (np.ndarray): A reference to the image.
        points (List[List[int]]): Points storage.
    """
    for x, y in points:
        cv.circle(img, (x,y), 5, RED, -1)
        
def take_reference_img_for_homography(vid_capture: cv.VideoCapture, reference_img_path: str):
    """Takes 0th frame from the video for using it as a reference image for creating homography.

    Args:
        vid_capture (cv.VideoCapture): Reference to the video
        reference_img_path (str): Path where to save the reference image.
    Returns:
        ret: whether the frame was succesfully read
        frame: 0th frame
    """
    ret, frame = vid_capture.read()
    if ret:
        cv.imwrite(reference_img_path, frame)
    else:
        print("Could not take reference img")
    return ret, frame


def squash_detections(path_to_detections: str, H: np.ndarray):
    """Squashes detections from the bounding box to the one point and transforms them using the homography matrix.
    Args:
        path_to_detections (str): 
        H (np.ndarray): Homography matrix.
    """
    start_time = time.time()
    storage = dict()
    last_frame_id = sys.maxsize
    detections, objects = [], []
    scaler_homo_func = lambda row: [int(row[0] / row[2]), int(row[1] / row[2]), 1.0]
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
                assert detections.shape[0] == len(objects)
                storage[last_frame_id] = (detections, objects)
                detections, objects = [], []
            detections.append([bb_left + 0.5 * bb_width, bb_top + bb_height, 1])
            objects.append(object_id)
            last_frame_id = frame_id
        print(f"Reading: {frame_id} frames took: {time.time() - start_time}s")
        return storage

    
def play_analysis(path_to_pitch: str, path_to_video: str, path_to_ref_img: str, path_to_detections: str):
    """ Two videos are being shown. One real which shows football match and the other one on which detections are being shown.
        Detections are drawn on a pitch image. Detections are mapped by a frame id. This is the current setup in which we first collect whole video and all detections by a tracker and then use this program
        to analyze data. In the future this can maybe be optimized so everything is being run online.
    Args:
        detections (np.array): Of shape N*3 where N is the total number of detections.
        detections_info (List[Tuple[int, int]]): List of tuples where each tuple contains information about frame id and object id.
        pitch_img (np.ndarray): Pitch image on which players will be drawn.
    """
    pitch_img = cv.imread(path_to_pitch, -1)
    mode = "full"
    # Setup video reading
    detections_vid_capture = cv.VideoCapture(path_to_video)
    reference_ret, reference_frame = take_reference_img_for_homography(detections_vid_capture, path_to_ref_img)
    H = visualize(src_points, dst_points, reference_frame, reference_frame.copy(), dst_img, dst_img_copy)
    print(f"H: {H:}")
    detections_storage = squash_detections(path_to_detections, H)
    # Setup window for showing match
    print(f"FPS: {detections_vid_capture.get(5)}")
    print(f"Frame count: {detections_vid_capture.get(7)}")
    full_screen_on_monitor(VIDEO_WINDOW)
    # Window for detections
    cv.namedWindow(DETECTIONS_WINDOW)
    monitor_info = get_offset_to_second_monitor()
    x_coord_det, y_coord_det = int(monitor_info[0] + 0.5 * monitor_info[2]), int(monitor_info[1] + 0.75 * monitor_info[3]) 
    cv.moveWindow(DETECTIONS_WINDOW, x_coord_det, y_coord_det); # where to put the window


    # Last frame id and detections per frame are used for playing birds-eye view
    consumed_first = False
    start_time = time.time()
    camera_frames = 0
    for frame_id, (detections_per_frame, object_ids) in detections_storage.items():
        k = cv.waitKey(1) & 0xFF
        # Playing birds-eye view
        frame_img = pitch_img.copy()
        print(f"Num detection in frame {frame_id}: {len(detections_per_frame)}")
        for frame_detection in detections_per_frame:
            cv.circle(frame_img, (int(frame_detection[0]), int(frame_detection[1])), 5, RED, -1)
        cv.imshow(DETECTIONS_WINDOW, frame_img)
        # Playing video with detections
        if not consumed_first and reference_ret:
            cv.imshow(VIDEO_WINDOW, reference_frame)
            consumed_first = True
        else:
            ret, video_frame = detections_vid_capture.read()
            if ret:
                cv.imshow(VIDEO_WINDOW, video_frame)
        camera_frames += 1
        # This will destroy all current windows being shown 
        if k == ord('f'):
            if mode == "full":
                print("Changing to resized")
                cv.setWindowProperty(VIDEO_WINDOW, cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
                mode = "resize"
            elif mode == "resize":
                print("Changing to full")
                cv.setWindowProperty(VIDEO_WINDOW, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                mode = "full"
        elif k == ord('q'):
            break
        
    print(f"Real FPS: {camera_frames / (time.time() - start_time):.2f}")
    print("Finished playing the video")
    detections_vid_capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(prog="FAPTRAS: Football analysis player tracking software", description="Software used for tracking players and generating statistics based on it.")
    parser.add_argument("--workdir", type=str, required=True, help="Working directory for saving data.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to the original video.")
    parser.add_argument("--detections-video-path", type=str, required=False, help="Path to the video with detections.")
    parser.add_argument("--detections-path", type=str, required=True, help="Path to the txt file with detections in MOT format.")
    parser.add_argument("--pitch-path", type=str, required=True, help="Path to the pitch image for visualizing in bird's eye mode.")
    args = parser.parse_args()
    
    # Path to images
    # Setup homography
    src_points, dst_points = [], []
    dst_img, dst_img_copy = read_image(args.pitch_path)


    #  H = np.array([[4.96880829e-03, 1.38109006e+00, -9.50897851e+01],
             #    [-3.33220488e-01, 4.19221488e-01, 9.04820655e+02],
             #    [1.35525043e-04, 2.46362905e-03, 1.00000000e+00]])

    ref_img = args.workdir + "/ref_img.jpg"
    print(f"Ref img: {ref_img}")
    
    play_analysis(args.pitch_path, args.video_path, ref_img, args.detections_path)