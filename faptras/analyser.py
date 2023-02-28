from typing import List, Tuple, Dict
import time
import argparse
import os
import json

import numpy as np
import cv2 as cv

from monitor_utils import get_offset_to_second_monitor
import view
import analytics_viewer
from pitch import Pitch
from match import Match
import config
import constants
import utils
import homography
import thread_prompter
import resolve_helpers as resolve_helpers

prompter = thread_prompter.ThreadWithReturnValue()

def play_visualizations(view_: view.View, pitch: Pitch, match: Match, detections_storage, pitch_img, detections_vid_capture, analytics: analytics_viewer.AnalyticsViewer, resolving_positions_cache: dict = None):
    """Plays visualization of both, real video and video created with the usage of homography.

    Args:
        view_ (view.View): A reference to the view object.
        pitch (Pitch): A reference to the pitch.
        detections_storage (_type_): Information needed to run visualization obtained by squash_detections function.
        pitch_img (_type_): A reference to the 2D pitch image.
        detections_vid_capture (_type_): Reference to the original video.

    Returns:
        status, resolving_cache: if status is True it means that visualization should be stopped. Resolving cache is dict. If None is given, cache will be filled and
            upon returning in the next method, flushed in the pickle format.
    """
    # Initialize view, set the timer
    start_time = time.time()
    
    cache_resolving = True
    if resolving_positions_cache is None:
        resolving_positions_cache = dict()
        cache_resolving = False
    
    resolve_helper = resolve_helpers.LastNFramesHelper(250, view_)   
        
    # Run visualizations
    for frame_id, (detections_per_frame, bb_info, object_ids) in detections_storage.items(): 
        # arr = [bb_info[0][0] + 0.5 * bb_info[0][2], bb_info[0][1] + bb_info[0][3], 1]
        # arr = H@np.array(arr)
        # print([arr[0] / arr[2], arr[1] / arr[2]], detections_per_frame[0])
        # Playing birds-eye view
        frame_img = pitch_img.copy()
        # Inefficient because we are creating copy of the list
        detections_in_pitch, bb_info_in_pitch, object_ids_in_pitch = pitch.get_objects_within(detections_per_frame, bb_info, object_ids)
        # Real video
        _, video_frame = detections_vid_capture.read()
        # Check whether we will have to deal with new object in this frame
        new_object_detections, new_object_bb_info, new_obj_ids = match.get_new_objects(bb_info_in_pitch, object_ids_in_pitch, detections_in_pitch)
        # Set is necessary but shouldn't be a problem since all new object ids should be different
        existing_ids_in_frame = [x for x in object_ids_in_pitch if x not in new_obj_ids]  # All ids that are shown in the frame and that are known from before
        for detection_info_new_id, bb_info_new_id, new_obj_id in zip(new_object_detections, new_object_bb_info, new_obj_ids):
            if cache_resolving:
                action = int(resolving_positions_cache[new_obj_id])
            else:
                print(f"New object id: {new_obj_id}")
                new_frame = video_frame.copy()
                prompter.set_execution_config(constants.prompt_input)
                view.View.box_label(new_frame, bb_info_new_id, constants.BLACK, new_obj_id)
                view.View.show_img_while_not_killed(constants.VIDEO_WINDOW, new_frame)
                action = int(prompter.value)
            match.resolve_user_action(action, new_obj_id, detection_info_new_id, resolving_positions_cache, new_obj_ids, existing_ids_in_frame, resolve_helper, prompter)    
        # Wait for the new key
        k = cv.waitKey(1) & 0xFF
        
        # Process per detection
        for i, frame_detection in enumerate(detections_in_pitch):
            id_to_show = str(object_ids_in_pitch[i])
            team1_player = match.team1.get_player(object_ids_in_pitch[i])
            team2_player = match.team2.get_player(object_ids_in_pitch[i])
            person = None
            if object_ids_in_pitch[i] in match.referee.ids:
                person = match.referee
                person_color = match.referee.color
            elif team1_player is not None:
                person = team1_player
                person_color, id_to_show = match.team1.color, str(team1_player.label)
            elif team2_player is not None:
                person = team2_player
                person_color, id_to_show = match.team2.color, str(team2_player.label)
            elif object_ids_in_pitch[i] not in match.ignore_ids:  # validation flag
                print(f"New object: {object_ids_in_pitch[i]}")
                os._exit(-1)
            
            if object_ids_in_pitch[i] not in match.ignore_ids:
                frame_detection_int = utils.to_tuple_int(frame_detection)
                view_.draw_person(frame_img, id_to_show, frame_detection_int, person_color)
                view.View.box_label(video_frame, bb_info_in_pitch[i], person_color, id_to_show)
                # If we are drawing the object, it means we can do analytics
                if (frame_id + 1) % analytics.run_estimation_frames == 0:
                    person.update_total_run(pitch.pixel_to_meters_positions(frame_detection_int))
                
        # Display
        cv.imshow(constants.DETECTIONS_WINDOW, frame_img)
        resolve_helper.storage.append(frame_img)
        cv.imshow(constants.VIDEO_WINDOW, video_frame)
        
        # Handle key-press
        utils.check_kill(k)
        if k == ord('f'):
            # Switch between normal and full mode
            view_.switch_screen_mode()
        elif k == ord('s'):
            # Switch drawing mode between circles and ids
            view_.switch_draw_mode()
        elif k == ord('p'):
            # Pause visualization
            utils.pause()
        elif k == ord('r'):
            # Show analytics
            analytics.show_player_run_table(match)
        elif k == ord('q'):
            # Quit visualization
            analytics.show_player_run_table(match)
            return True, resolving_positions_cache
        
    print(f"Real FPS: {frame_id / (time.time() - start_time):.2f}")
    return False, resolving_positions_cache
    
def play_analysis(view_: view.View, pitch: Pitch, path_to_pitch: str, path_to_video: str, path_to_ref_img: str, path_to_detections: str, 
                  cache_homography: bool, cache_initial_positions: bool, cache_resolving: bool):
    """ Two videos are being shown. One real which shows football match and the other one on which detections are being shown.
        Detections are drawn on a pitch image. Detections are mapped by a frame id. This is the current setup in which we first collect whole video and all detections by a tracker and then use this program
        to analyze data. In the future this can maybe be optimized so everything is being run online.
    Args:
    """
    pitch_img = cv.imread(pitch.img_path, -1) 
    # Setup video reading
    detections_vid_capture = cv.VideoCapture(path_to_video)
    num_frames = int(detections_vid_capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps_rate = int(detections_vid_capture.get(cv.CAP_PROP_FPS))
    analytics = analytics_viewer.AnalyticsViewer(int(num_frames / fps_rate)) # sample every 1s
    extracted_file_name = utils.get_file_name(path_to_video)
    homo_file = config.PATH_TO_HOMOGRAPHY_MATRICES + extracted_file_name + ".npy"
    player_cache_file = config.PATH_TO_INITIAL_PLAYER_POSITIONS + extracted_file_name + ".txt"
    resolving_positions_cache_file  = config.PATH_TO_RESOLVING_POSITIONS + extracted_file_name + ".json"
    if cache_homography:
        _, reference_frame = detections_vid_capture.read()
        try:
            H = np.load(homo_file)
            print(f"Using cached homography matrix...")
        except AttributeError | FileNotFoundError:
            print(f"Loading homography matrix failed, you will have to id manually...")
            reference_frame, H = homography.do_homography(view_, detections_vid_capture, path_to_ref_img, path_to_pitch, homo_file)
    else:
        reference_frame, H = homography.do_homography(view_, detections_vid_capture, path_to_ref_img, path_to_pitch, homo_file)
    print(f"H: {H:}")
    detections_storage = utils.squash_detections(path_to_detections, H)
    
    if cache_resolving:
        try:
            with open(resolving_positions_cache_file, "r") as resolving_positions_cache_f:
                resolving_positions_cache = json.load(resolving_positions_cache_f)
                resolving_positions_cache = dict(map(lambda x: (int(x[0]), int(x[1])), resolving_positions_cache.items()))
        except Exception as e:
            print(f"Fallback to manual resolving {e}")
            resolving_positions_cache = None
            cache_resolving = False
    else:
        resolving_positions_cache = None
    
    # Initialize match part 1
    sorted_keys = sorted(detections_storage.keys())  # sort by frame_id
    frame_detections0, bb_info0, object_ids0 = detections_storage[sorted_keys[0]]  # detections and object_ids in the 0th frame
    detections_storage.pop(sorted_keys[0]) # Don't visualize the first frame
    frame_detections0, bb_info0, object_ids0 = pitch.get_objects_within(frame_detections0, bb_info0, object_ids0)
    if cache_initial_positions: 
        match = Match.cache_team_resolution(pitch, player_cache_file)
        if not match:
            match = Match.user_team_resolution(pitch, object_ids0, frame_detections0, bb_info0, reference_frame, constants.VIDEO_WINDOW, player_cache_file, prompter)
            print("User needs to resolve initial setting")
        else:
            print("Using cached initial setting")
    else:
        match = Match.user_team_resolution(pitch, object_ids0, frame_detections0, bb_info0, reference_frame, constants.VIDEO_WINDOW, player_cache_file, prompter)
        print("User needs to resolve initial setting")

    # Window for detections
    cv.namedWindow(constants.DETECTIONS_WINDOW)
    monitor_info = get_offset_to_second_monitor()
    x_coord_det, y_coord_det = int(monitor_info[0] + 0.5 * monitor_info[2]), int(monitor_info[1] + 0.75 * monitor_info[3]) 
    cv.moveWindow(constants.DETECTIONS_WINDOW, x_coord_det, y_coord_det); # where to put the window
    view.View.full_screen_on_monitor(constants.VIDEO_WINDOW)
    # Run visualizations
    while True: 
        status, resolving_positions_cache = play_visualizations(view_, pitch, match, detections_storage, pitch_img, detections_vid_capture, analytics, resolving_positions_cache)
        # Restart the video if you didn't get any input
        if not cache_resolving and len(resolving_positions_cache) != 0:
           with open(resolving_positions_cache_file, "w") as resolving_positions_file:
                json.dump(resolving_positions_cache, resolving_positions_file, indent=2) 
        detections_vid_capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        if status:
            break
    
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
    parser.add_argument("--pitch-length", type=int, required=False, default=105, help="Pitch length in meters")
    parser.add_argument("--pitch-width", type=int, required=False, default=68, help="Pitch width in meters")
    parser.add_argument("--cache-homography", required=False, default=False, action="store_true", help="If set to True, the program will try to reuse the existing homography matrix. If flag set to True\
                        but there is no cached homography matrix for this video, the application will prompt you to enter it. ")
    parser.add_argument("--cache-initial-positions", required=False, default=False, action="store_true", help="If set to True, the program will try to reuse the existing information about players initial positions \
                        so that if there are players around the center, the user doesn't need to input to which team does the player belong")
    parser.add_argument("--cache-resolving", required=False, default=False, action="store_true", help="Whether to cache user resolving ids throughout the match")
    args = parser.parse_args()
    
    print(args.pitch_length, args.pitch_width)
    pitch = Pitch.load_pitch(args.pitch_path, args.pitch_length, args.pitch_width)
    # Setup view
    view_ = view.View(view.ViewMode.NORMAL)
    # view_.full_screen_on_monitor(constants.VIDEO_WINDOW)
    ref_img = args.workdir + "/ref_img.jpg"
    if not args.cache_homography:
        args.cache_initial_positions = False
        args.cache_resolving = False
        
    play_analysis(view_, pitch, args.pitch_path, args.video_path, ref_img, args.detections_path, args.cache_homography, args.cache_initial_positions, args.cache_resolving)
    # Stop the prompting thread
    prompter.running = False
    prompter.join()