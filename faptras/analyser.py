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
from person import Referee, Player, Person
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

def get_new_objects(match: Match, bb_infos: Tuple[int, int, int, int], object_ids: List[int]) -> List[int]:
    # TODO: write it in the functional manner
    new_objects: List[Tuple[int, int, int, int], int] = []
    for i, obj_id in enumerate(object_ids):
        if match.find_player_with_id(obj_id) is None:
            new_objects.append((bb_infos[i], obj_id))
    return new_objects

def create_new_player(match: Match) -> None:
    """Creates new player when the user cannot recognize it from the existing ones. The method asks user to provide details about the name and jersey number.

    Args:
        player_id (int): player id
    """
    print("Enter new player's team name: ")
    team = input()
    print("Enter new player's name: ")
    name = input()
    print("Enter player's jersey number: ")
    jersey_num = int(input())
    match.max_id += 1
    if team == match.team1.name:
        match.team1.players.append( Player(name, jersey_num, match.max_id) )
    elif team == match.team2.name:
        match.team2.players.append( Player(name, jersey_num, match.max_id) )

        

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
    for frame_id, (detections_per_frame, bb_info, object_ids) in detections_storage.items(): 
        # Playing birds-eye view
        frame_img = pitch_img.copy()
        detections_in_pitch = np.array(list(filter(lambda frame_detection: not is_detection_outside(pitch, frame_detection[0], frame_detection[1]), detections_per_frame)))
        # Real video
        _, video_frame = detections_vid_capture.read()
        # Check whether we will have to deal with new object in this frame
        new_object_ids = get_new_objects(match, bb_info, object_ids)
        for bb_info_new_id, new_obj_id in new_object_ids:
            print(f"New object id: {new_obj_id}")
            new_frame = video_frame.copy()
            # For now let's just extract the first element
            print(constants.prompt_input)
            view.View.box_label(new_frame, bb_info_new_id, constants.BLACK, new_obj_id)
            while True:
                k = cv.waitKey(1) & 0xFF
                cv.imshow(constants.VIDEO_WINDOW, new_frame)
                if k == ord('q'):
                    print(f"Exit from the new object detections")
                    break
            user_enter = input()
            if not user_enter:
                create_new_player(match)
            elif int(user_enter) == 0:
                match.referee.ids.append(new_obj_id)
            elif int(user_enter) != -1:
                ex_player = match.find_player_with_id(int(user_enter))
                print(f"Found ex player with ids: {ex_player.ids}")
                if not ex_player:  # if there is no player with this enter, repeat the procedure as when the user entered empty string
                    create_new_player(match)
                else:
                    ex_player.ids.append(new_obj_id)
        
        # Wait for the new key
        k = cv.waitKey(1) & 0xFF
        
        # Process per detection
        for i, frame_detection in enumerate(detections_in_pitch):
            if object_ids[i] in match.referee.ids:
                person_color = match.referee.color
            elif match.team1.get_player(object_ids[i]) is not None:
                person_color = match.team1.color
            elif match.team2.get_player(object_ids[i]) is not None:
                person_color = match.team2.color
            else:
                print(f"Error new object")
            view.View.draw_person(frame_img, str(object_ids[i]),(int(frame_detection[0]), int(frame_detection[1])), person_color)
            view.View.box_label(video_frame, bb_info[i], person_color, object_ids[i])
            
        # Display
        cv.imshow(constants.DETECTIONS_WINDOW, frame_img)
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
        # H, referee_id = calculate_homography(view_, src_points, dst_points, reference_frame, reference_frame.copy(), dst_img, dst_img_copy)
        print(f"H: {H:}")
    except:
        print(f"Couldn't create homography matrix, exiting from the program...")
        exit(-1)
    detections_storage = squash_detections(path_to_detections, H)
    
    # Initialize match part 1
    sorted_keys = sorted(detections_storage.keys())  # sort by frame_id
    frame_detections0, _, object_ids0 = detections_storage[sorted_keys[0]]  # detections and object_ids in the 0th frame
    detections_storage.pop(sorted_keys[0]) # Don't visualize the first frame
    detections_in_pitch0 = np.array(list(filter(lambda frame_detection: not is_detection_outside(pitch, frame_detection[0], frame_detection[1]), frame_detections0)))  # get only detections in the pitch
    # Initialize match part 2
    match = Match.initialize_match(pitch, detections_in_pitch0, object_ids0, referee_id)
    
    # Setup window for showing match
    print(f"FPS: {detections_vid_capture.get(5)}")
    print(f"Frame count: {detections_vid_capture.get(8)}")
    
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
    view_ = view.View(view.ViewMode.NORMAL)
    # view_.full_screen_on_monitor(constants.VIDEO_WINDOW)

    
    ref_img = args.workdir + "/ref_img.jpg"
    print(f"Ref img: {ref_img}")
    
    play_analysis(view_, pitch, args.video_path, ref_img, args.detections_path)