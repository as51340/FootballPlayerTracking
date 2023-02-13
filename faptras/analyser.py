from typing import List, Tuple, Dict
import os
from collections import defaultdict
import time
import sys
import argparse

import numpy as np
import cv2 as cv

from monitor_utils import get_offset_to_second_monitor
import view
from pitch import Pitch, PitchOrientation
# from team_classification import TeamClassificator, DBSCANTeamClassificator, KMeansTeamClassificator, get_bounding_boxes
from match import Match
from team import Team
import config
import constants


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

def calculate_homography(view_: view.View, src_points: List[List[int]], dst_points: List[List[int]], src_img: np.ndarray, src_img_copy: np.ndarray, dst_img: np.ndarray, dst_img_copy: np.ndarray):
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
    view.View.full_screen_on_monitor(constants.SRC_WINDOW)
    cv.setMouseCallback(constants.SRC_WINDOW, view_._select_points_wrapper, (constants.SRC_WINDOW, src_points, src_img_copy))
    # Destination window setup
    cv.namedWindow(constants.DST_WINDOW)
    cv.setMouseCallback(constants.DST_WINDOW, view_._select_points_wrapper, (constants.DST_WINDOW, dst_points, dst_img_copy))
    referee_id = -1
    while(1):
        # Keep showing copy of the image because of the circles drawn
        cv.imshow(constants.SRC_WINDOW, src_img_copy)
        cv.imshow(constants.DST_WINDOW, dst_img_copy)
        k = cv.waitKey(1) & 0xFF
        if k == ord('h'):
            print('create plan view')
            plan_view, H, _ = get_plan_view(src_img, dst_img, src_points, dst_points)
            print(f"Please insert referee id: ")
            referee_id = int(input())
            print(f"Referee has the initial id: {referee_id}")
            cv.imshow("plan view", plan_view)
        elif k == ord('d') and view_.last_clicked_windows:
            undo_window = view_.last_clicked_windows.pop()
            if undo_window == constants.SRC_WINDOW and src_points:
                src_points.pop()
                # Reinitialize image
                src_img_copy = src_img.copy()
                cv.setMouseCallback(constants.SRC_WINDOW, view_._select_points_wrapper, (constants.SRC_WINDOW, src_points, src_img_copy))
                view_.draw_old_circles(src_img_copy, src_points)
            elif undo_window == constants.DST_WINDOW and dst_points:
                dst_points.pop()
                # Reinitialize image
                dst_img_copy = dst_img.copy()
                cv.setMouseCallback(constants.DST_WINDOW, view_._select_points_wrapper, (constants.DST_WINDOW, dst_points, dst_img_copy))
                view_.draw_old_circles(dst_img_copy, dst_points)
        elif k == ord("s"):
            print(f"Source points: {src_points}")
            print(f"Dest points: {dst_points}")
        elif k == ord('q'):
            cv.destroyAllWindows()
            return H, referee_id

def take_reference_img_for_homography(vid_capture: cv.VideoCapture, reference_img_path: str):
    """Takes 0th frame from the video for using it as a reference image for creating homography.

    Args:
        vid_capture (cv.VideoCapture): Reference to the video
        reference_img_path (str): Path where to save the reference image.
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
    
def is_detection_outside(pitch: Pitch, detection_x_coord: int, detection_y_coord: int) -> bool:
    """Checks whether the detection is outside of the pitch.

    Args:
        pitch (Pitch): A reference to the pitch.
        detection_x_coord (int): x_coordinate in 2D pitch.
        detection_y_coord (int): y_coordinate in 2D pitch.

    Returns:
        bool: true if it is outside of the pitch.
    """
    if detection_x_coord < pitch.upper_left_corner[0] or detection_y_coord < pitch.upper_left_corner[1] or \
        detection_x_coord > pitch.upper_right_corner[0] or detection_y_coord > pitch.down_right_corner[1]:
        return True
    return False

def play_visualizations(pitch: Pitch, match: Match, detections_storage, pitch_img, detections_vid_capture):
    """Plays visualization of both, real video and video created with the usage of homography.

    Args:
        pitch (Pitch): _description_
        detections_storage (_type_): _description_
        pitch_img (_type_): _description_
        reference_frame (_type_): _description_
        detections_vid_capture (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Initialize view, set the timer
    start_time = time.time()
    
    # Run visualizations
    for frame_id, (detections_per_frame, _, object_ids) in detections_storage.items():
        k = cv.waitKey(1) & 0xFF
        
        # Playing birds-eye view
        frame_img = pitch_img.copy()
        detections_in_pitch = np.array(list(filter(lambda frame_detection: not is_detection_outside(pitch, frame_detection[0], frame_detection[1]), detections_per_frame)))
        print(f"Num detection in frame {frame_id}: {len(detections_in_pitch)}")
        for i, frame_detection in enumerate(detections_in_pitch):
            if object_ids[i] == match.referee_id:
                person_color = match.referee_color
            elif object_ids[i] in match.team1.players:
                person_color = match.team1.color
            elif object_ids[i] in match.team2.players:
                person_color = match.team2.color
            else:
                # TODO: prompt the user to solve algorithm's confusion
                print(f"New object identified with id: {object_ids[i]}")
                exit(-1) 
            view.draw_person(frame_img, str(object_ids[i]),(int(frame_detection[0]), int(frame_detection[1])), person_color)
        cv.imshow(constants.DETECTIONS_WINDOW, frame_img)
        
        # Real video
        _, video_frame = detections_vid_capture.read()
        # bboxes, max_height, max_width = get_bounding_boxes(bb_info, video_frame)
        # team_classificator.classify_persons_into_categories(bboxes, max_height, max_width)
        cv.imshow(constants.VIDEO_WINDOW, video_frame)
        
        # Handle key-press
        if k == ord('f'):
            view_mode = view.handle_screen_view_mode(view_mode)
        elif k == ord('q'):
            return True
        
    print(f"Real FPS: {frame_id / (time.time() - start_time):.2f}")
    return False
    
def play_analysis(view_: view.View, pitch: Pitch, path_to_video: str, path_to_ref_img: str, path_to_detections: str):
    """ Two videos are being shown. One real which shows football match and the other one on which detections are being shown.
        Detections are drawn on a pitch image. Detections are mapped by a frame id. This is the current setup in which we first collect whole video and all detections by a tracker and then use this program
        to analyze data. In the future this can maybe be optimized so everything is being run online.
    Args:
    """
    pitch_img = cv.imread(pitch.img_path, -1)
    
    # Setup video reading
    detections_vid_capture = cv.VideoCapture(path_to_video)
    _, reference_frame = take_reference_img_for_homography(detections_vid_capture, path_to_ref_img)
    
    # Create homography
    src_points, dst_points = [], []
    dst_img, dst_img_copy = view.View.read_image(args.pitch_path)
    try:
        H = np.array([[ 4.22432222e-01, 1.17147500e+00, -4.66582423e+02],
                [-1.71746715e-02, 2.59856556e+00, -1.85840310e+02],
                [-4.68413825e-05, 4.99413200e-03, 1.00000000e+00]])
        referee_id = 23
        H, referee_id = calculate_homography(view_, src_points, dst_points, reference_frame, reference_frame.copy(), dst_img, dst_img_copy)
        print(f"H: {H:}")
    except:
        print(f"Couldn't create homography matrix, exiting from the program...")
        exit(-1)
    detections_storage = squash_detections(path_to_detections, H)
    
    # Initialize match
    sorted_keys = sorted(detections_storage.keys())
    frame_detections0, _, object_ids0 = detections_storage[sorted_keys[0]]
    detections_storage.pop(sorted_keys[0]) # Don't visualize the first frame
    detections_in_pitch0 = np.array(list(filter(lambda frame_detection: not is_detection_outside(pitch, frame_detection[0], frame_detection[1]), frame_detections0)))
    match = Match.initialize_match(pitch, detections_in_pitch0, object_ids0, referee_id)
    
    # Setup window for showing match
    print(f"FPS: {detections_vid_capture.get(5)}")
    print(f"Frame count: {detections_vid_capture.get(7)}")
    view.full_screen_on_monitor(constants.VIDEO_WINDOW)
    
    # Window for detections
    cv.namedWindow(constants.DETECTIONS_WINDOW)
    monitor_info = get_offset_to_second_monitor()
    x_coord_det, y_coord_det = int(monitor_info[0] + 0.5 * monitor_info[2]), int(monitor_info[1] + 0.75 * monitor_info[3]) 
    cv.moveWindow(constants.DETECTIONS_WINDOW, x_coord_det, y_coord_det); # where to put the window
    
    # Run visualizations
    while not play_visualizations(pitch, match, detections_storage, pitch_img, detections_vid_capture):
        # Restart the video if you didn't get any input
        print("Video loop ON")
        detections_vid_capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        
    # Postprocess
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
    
    pitch = Pitch.load_pitch(args.pitch_path)
    view_ = view.View()
    
    ref_img = args.workdir + "/ref_img.jpg"
    print(f"Ref img: {ref_img}")
    
    play_analysis(view_, pitch, args.video_path, ref_img, args.detections_path)