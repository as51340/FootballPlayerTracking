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
import utils

def check_kill(k):
    if k == ord('k'):
        exit(-1)

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

def do_homography(detections_vid_capture, path_to_ref_img: str, path_to_pitch: str, homo_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper function that does all necessary preparation for calling calculate_homography function.

    Args:
        detections_vid_capture (_type_): reference to the video
        path_to_ref_img (str): 
        path_to_pitch (str): 
        homo_file (str): path where the homography file is saved.

    Returns:
        Tuple[np.ndarray, np.ndarray]: reference frame = 0th frame from th video and homography matrix.
    """
    _, reference_frame = take_reference_img_for_homography(detections_vid_capture, path_to_ref_img)
    # Create homography
    H = calculate_homography(view_, reference_frame, path_to_pitch)
    np.save(homo_file, H)
    return reference_frame, H

def calculate_homography(view_: view.View, src_img: np.ndarray, path_to_pitch: str):
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
    src_points, dst_points = [], []
    src_img_copy: np.ndarray = src_img.copy()
    dst_img = cv.imread(path_to_pitch, -1)
    dst_img_copy: np.ndarray = dst_img.copy()
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
            cv.imshow("plan view", plan_view)
        elif k == ord('d') and view_.last_clicked_windows:
            undo_window = view_.last_clicked_windows.pop()
            if undo_window == constants.SRC_WINDOW and src_points:
                src_points.pop()
                # Reinitialize image
                src_img_copy = src_img.copy()
                cv.setMouseCallback(constants.SRC_WINDOW, view_._select_points_wrapper, (constants.SRC_WINDOW, src_points, src_img_copy))
                view.View.draw_2d_objects(src_img_copy, src_points)
            elif undo_window == constants.DST_WINDOW and dst_points:
                dst_points.pop()
                # Reinitialize image
                dst_img_copy = dst_img.copy()
                cv.setMouseCallback(constants.DST_WINDOW, view_._select_points_wrapper, (constants.DST_WINDOW, dst_points, dst_img_copy))
                view.View.draw_2d_objects(dst_img_copy, dst_points)
        elif k == ord("s"):
            print(f"Source points: {src_points}")
            print(f"Dest points: {dst_points}")
        elif k == ord('q'):
            cv.destroyAllWindows()
            return H

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
        if obj_id not in match.ignore_ids and match.find_person_with_id(obj_id) is None:
            new_objects.append((bb_infos[i], obj_id))
    return new_objects

def create_new_player(match: Match) -> None:
    """Creates new player when the user cannot recognize it from the existing ones. The method asks user to provide details about the name and jersey number.

    Args:
        player_id (int): player id
    """
    print("Enter new player's name: ")
    name = input()
    print("Enter player's jersey number: ")
    jersey_num = int(input())
    match.max_id += 1
    while True:
        print("Enter new player's team name: ")
        team = input()
        if team == match.team1.name:
            match.team1.players.append( Player(name, jersey_num, match.max_id) )
            break
        elif team == match.team2.name:
            match.team2.players.append( Player(name, jersey_num, match.max_id) )
            break
        else:
            print(f"Such team doesn't exist, in this match there are two teams: {match.team1.name} {match.team2.name}")

def resolve_team_helper(match: Match, team: str, player_id: int):
    """Adds player to the team based on the team name and returns True. If no such team exists in the match, the method returns false.

    Args:
        match (Match): A reference to the match.
        team (str): Team name
        player_id (int): Player id which needs to be added to some team.

    Returns:
        bool: True if team matches one of the teams' names, False otherwise.
    """
    if team.lower() == match.team1.name.lower():
        match.team1.players.append(Player("name", -1, player_id))
        return True
    elif team.lower() == match.team2.name.lower():
        match.team2.players.append(Player("name", -1, player_id))
        return True
    return False


def user_team_resolution(match: Match, uncertain_objs: List[Tuple[int, Tuple[int, int, int, int]]], img: np.ndarray, window: str, cache_file: str) -> None:
    """User manually decides about the team of the player shown on the image. If user inputs the team that is not part of the current match, the procedure is repeated for the same player.

    Args:
        match (Match): A reference to the match.
        uncertain_objs (List[Tuple[int, Tuple[int, int]]]): All objects that need to be resolved. First item in tuple is the object id that will be shown on the image. Second item are 
            bounding box coordinates. 
        img (np.ndarray): A reference to the image on which players will be drawn.
        window (str): Window name.
        
    """
    player_cache = dict()
    for uncertain_obj_id, bb_obj_info in uncertain_objs:
        show_frame = img.copy() 
        view.View.box_label(show_frame, bb_obj_info, constants.BLACK, uncertain_obj_id)
        while True:
            print(f"Please insert the team of the player shown on the screen: ")
            # new thread
            while True:
                k = cv.waitKey(1) & 0xFF
                cv.imshow(window, show_frame)
                if k == ord('q'):
                    break
                check_kill(k)
            team = input()
            print(f"Team: {team}")
            if not resolve_team_helper(match, team, uncertain_obj_id):
                print("Unknown team, please insert again...")
            else:
                player_cache[uncertain_obj_id] = team.lower()
                break
    with open(cache_file, "w") as file:
        for obj_id, team in player_cache.items():
            file.write(f"{obj_id}###{team}\n")
    
                
def cache_team_resolution(match: Match, uncertain_objs: List[Tuple[int, Tuple[int, int, int, int]]], cache_file: str) -> bool:
    """Resolves players' teams based on the cached file. This is used at the beginning, where the user needs to correctly classify players around the center.

    Args:
        match (Match): A reference to the match.
        uncertain_objs (List[int, Tuple[int, int, int, int]]): List of unresolved players' ids/
        cache_file (str): Path to the file where all information is being saved.

    Returns:
        bool: True if the process went ok and team could be found for every unresolved player, otherwise returns False. If the file doesn't exist, the process returns False.
    """
    players_data = dict()
    try:
        with open(cache_file, "r") as cache_file:
            while (line := cache_file.readline().rstrip()):
                line_data = line.split('###')
                players_data[int(line_data[0])] = line_data[1]
    except:
        return False
    
    success = True
    for uncertain_obj_id, _ in uncertain_objs:
        cached_team = players_data[uncertain_obj_id]
        if not resolve_team_helper(match, cached_team, uncertain_obj_id):
            success = False
            break    
    return success        

def get_objects_within_pitch(pitch: Pitch, detections, bb_info, object_ids) -> Tuple[Tuple[int, int], Tuple[int, int, int, int], List[int]]:
    """Returns detections (2D objects), bounding boxes and object_ids only of objects which are within the pitch boundary.

    Args:
        pitch (Pitch): A reference to the pitch.
        detections (_type_): 2D detections.
        bb_info (_type_): Bounding box information.
        object_ids (_type_): Objects ids

    Returns:
        _type_: _description_
    """
    detections_in_pitch, bb_info_ids, object_ids_in_pitch = [], [], []
    for i, detection in enumerate(detections):
        if not is_detection_outside(pitch, detection[0], detection[1]):
            detections_in_pitch.append(detection)
            bb_info_ids.append(bb_info[i])
            object_ids_in_pitch.append(object_ids[i])
    return detections_in_pitch, bb_info_ids, object_ids_in_pitch
    
def play_visualizations(view_: view.View, pitch: Pitch, match: Match, detections_storage, pitch_img, detections_vid_capture):
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
        # Inefficient as fuck because we are creating copy of the list
        detections_in_pitch, bb_info_in_pitch, object_ids_in_pitch = get_objects_within_pitch(pitch, detections_per_frame, bb_info, object_ids)
        # Real video
        _, video_frame = detections_vid_capture.read()
        # Check whether we will have to deal with new object in this frame
        new_object_ids = get_new_objects(match, bb_info_in_pitch, object_ids_in_pitch)
        for bb_info_new_id, new_obj_id in new_object_ids:
            print(f"New object id: {new_obj_id}")
            new_frame = video_frame.copy()
            # For now let's just extract the first element
            print(constants.prompt_input)
            view.View.box_label(new_frame, bb_info_new_id, constants.BLACK, new_obj_id)
            while True:
                k = cv.waitKey(1) & 0xFF
                cv.imshow(constants.VIDEO_WINDOW, new_frame)
                check_kill(k)
                if k == ord('q'):
                    break
            user_enter = input()
            if not user_enter:
                create_new_player(match)
            elif int(user_enter) == 0:
                match.referee.ids.append(new_obj_id)
            elif int(user_enter) == -1:
                match.ignore_ids.append(new_obj_id)  # from now on ignore this id
            else:
                ex_player = match.find_player_with_id(int(user_enter))
                if ex_player is not None:
                    print(f"Found ex player with ids: {ex_player.ids}")
                    ex_player.ids.append(new_obj_id)
                else:
                    create_new_player(match)
        
        # Wait for the new key
        k = cv.waitKey(1) & 0xFF
        
        # Process per detection
        for i, frame_detection in enumerate(detections_in_pitch):
            if object_ids_in_pitch[i] in match.referee.ids:
                person_color = match.referee.color
            elif match.team1.get_player(object_ids_in_pitch[i]) is not None:
                person_color = match.team1.color
            elif match.team2.get_player(object_ids_in_pitch[i]) is not None:
                person_color = match.team2.color
            elif object_ids_in_pitch[i] not in match.ignore_ids:  # validation flag
                print(f"Error new object")
                exit(-1)
            
            if object_ids_in_pitch[i] not in match.ignore_ids:
                view_.draw_person(frame_img, str(object_ids_in_pitch[i]),(int(frame_detection[0]), int(frame_detection[1])), person_color)
                view.View.box_label(video_frame, bb_info_in_pitch[i], person_color, object_ids_in_pitch[i])  # BUG, TODO, wrong indexing
            
        # Display
        cv.imshow(constants.DETECTIONS_WINDOW, frame_img)
        # bboxes, max_height, max_width = get_bounding_boxes(bb_info, video_frame)
        # team_classificator.classify_persons_into_categories(bboxes, max_height, max_width)
        cv.imshow(constants.VIDEO_WINDOW, video_frame)
        
        # Handle key-press
        if k == ord('f'):
            view_.switch_screen_mode()
        elif k == ord('s'):
            view_.switch_draw_mode()
        elif k == ord('q'):
            return True
        
    print(f"Real FPS: {frame_id / (time.time() - start_time):.2f}")
    return False
    
def play_analysis(view_: view.View, pitch: Pitch, path_to_pitch: str, path_to_video: str, path_to_ref_img: str, path_to_detections: str, team1_name: str, team2_name: str, 
                  cache_homography: bool, cache_initial_positions: bool):
    """ Two videos are being shown. One real which shows football match and the other one on which detections are being shown.
        Detections are drawn on a pitch image. Detections are mapped by a frame id. This is the current setup in which we first collect whole video and all detections by a tracker and then use this program
        to analyze data. In the future this can maybe be optimized so everything is being run online.
    Args:
    """
    pitch_img = cv.imread(pitch.img_path, -1)
    
    # Setup video reading
    detections_vid_capture = cv.VideoCapture(path_to_video)
    extracted_file_name = utils.get_file_name(path_to_video)
    homo_file = config.PATH_TO_HOMOGRAPHY_MATRICES + extracted_file_name + ".npy"
    player_cache_file = config.PATH_TO_INITIAL_PLAYER_POSITIONS + extracted_file_name + ".txt"
    if cache_homography:
        _, reference_frame = detections_vid_capture.read()
        try:
            H = np.load(homo_file)
            print(f"Using cached homography matrix...")
        except:
            print(f"Loading homography matrix failed, you will have to id manually...")
            reference_frame, H = do_homography(detections_vid_capture, path_to_ref_img, path_to_pitch, homo_file)
    else:
        reference_frame, H = do_homography(detections_vid_capture, path_to_ref_img, path_to_pitch, homo_file)
    print(f"H: {H:}")
    detections_storage = squash_detections(path_to_detections, H)
    
    # Initialize match part 1
    sorted_keys = sorted(detections_storage.keys())  # sort by frame_id
    frame_detections0, bb_info0, object_ids0 = detections_storage[sorted_keys[0]]  # detections and object_ids in the 0th frame
    detections_storage.pop(sorted_keys[0]) # Don't visualize the first frame
    frame_detections0, bb_info0, object_ids0 = get_objects_within_pitch(pitch, frame_detections0, bb_info0, object_ids0)
    # Initialize match part 2
    # First let the user decide about the referee id
    ref_reference_frame = reference_frame.copy()
    for i, bb_obj_info0 in enumerate(bb_info0):
        view.View.box_label(ref_reference_frame, bb_obj_info0, constants.BLACK, object_ids0[i])
    # Create real match
    print("Please enter referee id: ")
    while True:
        k = cv.waitKey(1) & 0xFF
        cv.imshow(constants.VIDEO_WINDOW, ref_reference_frame)
        check_kill(k)
        if k == ord('q'):
            break
    referee_id = int(input())
    print(f"Referee id: {referee_id}")
    # Create real match
    match, uncertain_objs = Match.initialize_match(pitch, frame_detections0, bb_info0, object_ids0, referee_id, team1_name, team2_name)
    if cache_initial_positions:
        if not cache_team_resolution(match, uncertain_objs, player_cache_file):
            user_team_resolution(match, uncertain_objs, reference_frame, constants.VIDEO_WINDOW, player_cache_file)
    else:
        user_team_resolution(match, uncertain_objs, reference_frame, constants.VIDEO_WINDOW, player_cache_file)
    
    # Window for detections
    cv.namedWindow(constants.DETECTIONS_WINDOW)
    monitor_info = get_offset_to_second_monitor()
    x_coord_det, y_coord_det = int(monitor_info[0] + 0.5 * monitor_info[2]), int(monitor_info[1] + 0.75 * monitor_info[3]) 
    cv.moveWindow(constants.DETECTIONS_WINDOW, x_coord_det, y_coord_det); # where to put the window
    
    # Run visualizations
    while not play_visualizations(view_, pitch, match, detections_storage, pitch_img, detections_vid_capture):
        # Restart the video if you didn't get any input
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
    parser.add_argument("--team1-name", type=str, required=True, help="First team's name")
    parser.add_argument("--team2-name", type=str, required=True, help="Second team's name")
    parser.add_argument("--cache-homography", required=False, default=False, action="store_true", help="If set to True, the program will try to reuse the existing homography matrix. If flag set to True\
                        but there is no cached homography matrix for this video, the application will prompt you to enter it. ")
    parser.add_argument("--cache-initial-positions", required=False, default=False, action="store_true", help="If set to True, the program will try to reuse the existing information about players initial positions \
                        so that if there are players around the center, the user doesn't need to input to which team does the player belong")
    args = parser.parse_args()
    
    pitch = Pitch.load_pitch(args.pitch_path)
    # Setup view
    view_ = view.View(view.ViewMode.NORMAL)
    # view_.full_screen_on_monitor(constants.VIDEO_WINDOW)

    ref_img = args.workdir + "/ref_img.jpg"
    print(f"Cached homography: {args.cache_homography}")
    
    play_analysis(view_, pitch, args.pitch_path, args.video_path, ref_img, args.detections_path, args.team1_name, args.team2_name, args.cache_homography, args.cache_initial_positions)