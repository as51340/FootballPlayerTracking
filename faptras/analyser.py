from typing import Dict
from collections import OrderedDict
import time
import argparse
import json
from enum import Enum

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
import ai_resolver
import sanity_checker
import keyboard_handler
from game_situations import GameSituations

prompter = thread_prompter.ThreadWithReturnValue()

VisualizationMode = Enum("VisualizationMode", ["PLAY", "SKIP"])


def end_visualizations(game_situations: GameSituations, writer_det, writer_orig, save_video: bool):
    if game_situations.needs_release:
        game_situations.video.release()
    if save_video:
        writer_det.release()
        writer_orig.release()
    pass


def play_visualizations(view_: view.View, pitch: Pitch, match: Match, detections_storage, pitch_img, detections_vid_capture,
                        fps_rate: int, writer_orig, writer_det, game_situations: GameSituations, save_video: bool, resolving_positions_cache: dict = None):
    """Plays visualization of both, real video and video created with the usage of homography.

    Args:
        view_ (view.View): A reference to the view object.
        pitch (Pitch): A reference to the pitch.
        detections_storage (_type_): Information needed to run visualization obtained by squash_detections function.
        pitch_img (_type_): A reference to the 2D pitch image.
        detections_vid_capture (_type_): Reference to the original video.
        fps_rate (int): FPS of the original video.
        writer_orig (_type_): A reference to the video writer for the original video.
        writer_det (_type_): A reference to the video writer for the 2D pitch video.
        game_situations (GameSituations): A reference to the game situations object.
        save_video (bool): Whether to save the video. If False, writer_orig and writer_det will be None.
        resolving_positions_cache (dict, optional): A reference to the resolving positions cache. Defaults to None.

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

    analytics_display = analytics_viewer.AnalyticsViewer()
    resolver = ai_resolver.Resolver(fps_rate)
    sanitizer = sanity_checker.SanityChecker()
    resolve_helper = resolve_helpers.LastNFramesHelper(250, view_)
    visualization_mode = VisualizationMode.PLAY

    # Run visualizations
    # it is ordered dict so order is preserved
    detections_storage = list(detections_storage.items())
    frame_index = 0
    while frame_index < len(detections_storage):
        frame_id, (detections_per_frame, bb_info,
                   object_ids, classes) = detections_storage[frame_index]
        # Real video
        _, video_frame = detections_vid_capture.read()

        # Playing birds-eye view
        frame_img_det = pitch_img.copy()

        # This should probably be somehow optimized
        detections_in_pitch, bb_info_in_pitch, objects_id_in_pitch, classes_in_pitch = sanitizer.clear_already_resolved(
            *pitch.get_objects_within(detections_per_frame, bb_info, object_ids, classes), resolving_positions_cache)
        # Check whether we will have to deal with new object in this frame
        # New objects can be only persons, not the ball
        new_objects_detection, new_objects_bb, new_objects_id, ball_id = match.get_new_objects(
            bb_info_in_pitch, objects_id_in_pitch, detections_in_pitch, classes_in_pitch)
        # All objects that can be resolved from before
        existing_objects_detection, existing_objects_bb, existing_objects_id = utils.get_existing_objects(
            detections_in_pitch, bb_info_in_pitch, objects_id_in_pitch, classes_in_pitch, new_objects_id)
        if ball_id is None:
            assert set(existing_objects_id).union(
                set(new_objects_id)) == set(objects_id_in_pitch)
        else:
            assert set(existing_objects_id).union(
                set(new_objects_id), set([ball_id])) == set(objects_id_in_pitch)

        # But if it is caching just try to get all ids from the cache
        if cache_resolving:
            for detection_info_new_id, bb_info_new_id, new_obj_id in zip(new_objects_detection, new_objects_bb, new_objects_id):
                old_player_id = int(resolving_positions_cache[new_obj_id])
                match.resolve_user_action(old_player_id, new_obj_id, detection_info_new_id,
                                          resolving_positions_cache, new_objects_id, existing_objects_id, resolve_helper, prompter)
        else:
            # Ask resolver for the help
            if len(new_objects_id):
                resolving_info: ai_resolver.ResolvingInfo = resolver.resolve(
                    pitch, match, new_objects_detection, new_objects_bb, new_objects_id, existing_objects_detection, existing_objects_bb, existing_objects_id, frame_id)
                # Resolve action for all resolved objects
                for i in range(len(resolving_info.resolved_ids)):
                    match.resolve_user_action(resolving_info.found_ids[i], resolving_info.resolved_ids[i], resolving_info.resolved_detections[
                                              i], resolving_positions_cache, new_objects_id, existing_objects_id, resolve_helper, prompter)
                # Do manual resolvement for all unresolved objects
                print(f"Starting manual resolvement in frame {frame_id}")
                if len(resolving_info.unresolved_ids):
                    print(f"Ids to resolve: {resolving_info.unresolved_ids}.")
                    print(
                        f"Missing players from the start: {resolving_info.unresolved_starting_ids}")
                assert len(set(resolving_info.unresolved_starting_ids).intersection(
                    set(existing_objects_id))) == 0
                # Draw known objects in the 2D space
                new_frame_det = pitch_img.copy()
                for j in range(len(existing_objects_id)):
                    _, existing_person_color, existing_id_to_show = match.get_info_for_drawing(
                        existing_objects_id[j])
                    view_.draw_2d_obj(new_frame_det, str(
                        existing_id_to_show), existing_objects_detection[j], existing_person_color, False)
                for i in range(len(resolving_info.unresolved_ids)):
                    unresolved_id_str = str(resolving_info.unresolved_ids[i])
                    print(f"Please resolve manually id {unresolved_id_str}")
                    new_frame_bb = video_frame.copy()
                    manual_frame_det = new_frame_det.copy()
                    # Draw unknown object in the 2D space
                    view_.draw_2d_obj(manual_frame_det, unresolved_id_str,
                                      resolving_info.unresolved_detections[i], constants.BLACK, False)
                    if i > 0:
                        resolve_helper.storage.pop()
                    resolve_helper.storage.append(manual_frame_det)
                    # Draw unknown object in the real video
                    view.View.box_label(
                        new_frame_bb, resolving_info.unresolved_bbs[i], constants.BLACK, unresolved_id_str)
                    prompter.set_execution_config(constants.prompt_input)
                    view.View.show_img_while_not_killed(
                        [constants.VIDEO_WINDOW, constants.DETECTIONS_WINDOW], [new_frame_bb, manual_frame_det])
                    action = int(prompter.value)
                    match.resolve_user_action(action, resolving_info.unresolved_ids[i], resolving_info.unresolved_detections[
                                              i], resolving_positions_cache, resolving_info.unresolved_ids, existing_objects_id, resolve_helper, prompter)
                if len(resolving_info.unresolved_ids):
                    resolve_helper.storage.pop()
                print()

        # Wait for the new key
        k = cv.waitKey(1) & 0xFF

        # Before visualizing, run sanity check
        sanitizer.check(objects_id_in_pitch)

        # Process per detection
        showing_ids = set()
         # TODO: Refactor this code
        for i, frame_detection in enumerate(detections_in_pitch):
            if objects_id_in_pitch[i] in match.ignore_ids:
                continue
            if classes_in_pitch[i] == constants.BALL_CLASS:
                id_to_show = "Ball"
                view_.draw_2d_obj(frame_img_det, id_to_show, utils.to_tuple_int(frame_detection), match.ball.color, True)
                view.View.box_label(
                    video_frame, bb_info_in_pitch[i], match.ball.color, "")
            else:
                person, person_color, id_to_show = match.get_info_for_drawing(
                    objects_id_in_pitch[i])
                person.last_seen_frame_id = frame_id

                # Validation step
                assert id_to_show not in showing_ids
                showing_ids.add(id_to_show)

                if objects_id_in_pitch[i] not in match.ignore_ids:
                    view_.draw_2d_obj(frame_img_det, id_to_show, utils.to_tuple_int(
                        frame_detection), person_color, False)
                    view.View.box_label(
                        video_frame, bb_info_in_pitch[i], person_color, id_to_show)
                    person.update_person_position(
                        frame_detection, pitch.pixel_to_meters_positions(frame_detection), frame_id)

        # This action needs to be done no matter the visualization mode
        if visualization_mode == VisualizationMode.PLAY:
            # Display
            cv.imshow(constants.DETECTIONS_WINDOW, frame_img_det)
            cv.imshow(constants.VIDEO_WINDOW, video_frame)

        resolve_helper.storage.append(frame_img_det)
        if save_video:
            writer_det.write(frame_img_det)
            writer_orig.write(video_frame)
        # Check whether we need to save some frame to the video
        if game_situations.needs_saving():
            game_situations.video.write(video_frame)

        # Handle key-press
        stop, seek_frame = keyboard_handler.handle_key_press(
            k, view_, analytics_display, pitch, match, fps_rate, frame_id, game_situations)
        if stop:
            break
        if seek_frame != 0:
            visualization_mode = VisualizationMode.SKIP
            frame_index = min(len(detections_storage)-1,
                              max(0, frame_index+seek_frame))
            detections_vid_capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
        else:
            visualization_mode = VisualizationMode.PLAY
            frame_index += 1

    print(f"Real FPS: {frame_id / (time.time() - start_time):.2f}")
    end_visualizations(game_situations, writer_det, writer_orig, save_video)
    # Show at the end running statistics
    # keyboard_handler.forward_analytics_calls([analytics_display.show_match_acc_summary, analytics_display.show_match_total_run, analytics_display.show_match_sprint_summary], pitch, match, fps_rate, 7)
    return resolving_positions_cache


def play_analysis(view_: view.View, pitch: Pitch, path_to_pitch: str, path_to_video: str, path_to_ref_img: str, path_to_detections: str,
                  cache_homography: bool, cache_initial_positions: bool, cache_resolving: bool, save_video: bool):
    """ Two videos are being shown. One real which shows football match and the other one on which detections are being shown.
        Detections are drawn on a pitch image. Detections are mapped by a frame id. This is the current setup in which we first collect whole video and all detections by a tracker and then use this program
        to analyze data. In the future this can maybe be optimized so everything is being run online.
    Args:
    """
    pitch_img = cv.imread(pitch.img_path, -1)
    # Setup video reading
    detections_vid_capture = cv.VideoCapture(path_to_video)
    fps_rate = int(detections_vid_capture.get(cv.CAP_PROP_FPS))
    # Setup video writing
    width_orig = int(detections_vid_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height_orig = int(detections_vid_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    game_situations = GameSituations(width_orig, height_orig, fps_rate)
    if save_video:
        write_orig = cv.VideoWriter('bboxes.mp4', cv.VideoWriter_fourcc(
            *'mp4v'), 30, (width_orig, height_orig))
        height_det, width_det, _ = pitch_img.shape
        write_det = cv.VideoWriter('detections.mp4', cv.VideoWriter_fourcc(
            *'mp4v'), 30, (width_det, height_det))
    else:
        write_orig, write_det = None, None

    # Prepare cache files
    extracted_file_name = utils.get_file_name(path_to_video)
    homo_file = config.PATH_TO_HOMOGRAPHY_MATRICES + extracted_file_name + ".npy"
    player_cache_file = config.PATH_TO_INITIAL_PLAYER_POSITIONS + \
        extracted_file_name + ".txt"
    resolving_positions_cache_file = config.PATH_TO_RESOLVING_POSITIONS + \
        extracted_file_name + ".json"

    # Prepare homography
    if cache_homography:
        _, reference_frame = detections_vid_capture.read()
        try:
            H = np.load(homo_file)
            print(f"Using cached homography matrix...")
        except AttributeError | FileNotFoundError:
            print(f"Loading homography matrix failed, you will have to id manually...")
            reference_frame, H = homography.do_homography(
                view_, detections_vid_capture, path_to_ref_img, path_to_pitch, homo_file)
    else:
        reference_frame, H = homography.do_homography(
            view_, detections_vid_capture, path_to_ref_img, path_to_pitch, homo_file)
    print(f"H: {H:}")

    # Prepare resolving cache
    if cache_resolving:
        try:
            with open(resolving_positions_cache_file, "r") as resolving_positions_cache_f:
                resolving_positions_cache = json.load(
                    resolving_positions_cache_f)
                resolving_positions_cache = dict(
                    map(lambda x: (float(x[0]), float(x[1])), resolving_positions_cache.items()))
        except Exception as e:
            print(f"Fallback to manual resolving {e}.")
            resolving_positions_cache = None
            cache_resolving = False
    else:
        resolving_positions_cache = None

    # Prepare initialization
    detections_storage: OrderedDict = utils.squash_detections(
        path_to_detections, H)  # ordered by the frame id
    _, (frame_detections1, bb_info1, object_ids1, classes) = detections_storage.popitem(
        last=False)  # detections and object_ids in the 1st frame
    frame_detections1, bb_info1, object_ids1, classes1 = pitch.get_objects_within(
        frame_detections1, bb_info1, object_ids1, classes)
    if cache_initial_positions:
        match = Match.cache_team_resolution(pitch, player_cache_file)
        if not match:
            print("User needs to resolve initial settings.")
            match = Match.user_team_resolution(
                pitch, object_ids1, frame_detections1, bb_info1, reference_frame, constants.VIDEO_WINDOW, player_cache_file, prompter)
        else:
            print("Using cached initial settings.")
    else:
        print("User needs to resolve initial setting.")
        match = Match.user_team_resolution(pitch, classes, object_ids1, frame_detections1, bb_info1,
                                           reference_frame, constants.VIDEO_WINDOW, player_cache_file, prompter)

    # Prepare visualizations
    cv.namedWindow(constants.DETECTIONS_WINDOW)
    monitor_info = get_offset_to_second_monitor()
    x_coord_det, y_coord_det = int(
        monitor_info[0] + 0.5 * monitor_info[2]), int(monitor_info[1] + 0.75 * monitor_info[3])
    cv.moveWindow(constants.DETECTIONS_WINDOW, x_coord_det,
                  y_coord_det)  # where to put the window
    view.View.full_screen_on_monitor(constants.VIDEO_WINDOW)
    resolving_positions_cache = play_visualizations(view_, pitch, match, detections_storage, pitch_img, detections_vid_capture,
                                                    fps_rate, write_orig, write_det, game_situations, save_video, resolving_positions_cache)
    if not cache_resolving and len(resolving_positions_cache) != 0:
        with open(resolving_positions_cache_file, "w") as resolving_positions_file:
            json.dump(resolving_positions_cache,
                      resolving_positions_file, indent=2)

    # Postprocess
    print("Finished playing the video")
    detections_vid_capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(prog="FAPTRAS: Football analysis player tracking software",
                                     description="Software used for tracking players and generating statistics based on it.")
    parser.add_argument("--workdir", type=str, required=True,
                        help="Working directory for saving data.")
    parser.add_argument("--video-path", type=str,
                        required=True, help="Path to the original video.")
    parser.add_argument("--detections-video-path", type=str,
                        required=False, help="Path to the video with detections.")
    parser.add_argument("--detections-path", type=str, required=True,
                        help="Path to the txt file with detections in MOT format.")
    parser.add_argument("--pitch-path", type=str, required=True,
                        help="Path to the pitch image for visualizing in bird's eye mode.")
    parser.add_argument("--pitch-length", type=int, required=False,
                        default=105, help="Pitch length in meters")
    parser.add_argument("--pitch-width", type=int, required=False,
                        default=68, help="Pitch width in meters")
    parser.add_argument("--cache-homography", required=False, default=False, action="store_true", help="If set to True, the program will try to reuse the existing homography matrix. If flag set to True\
                        but there is no cached homography matrix for this video, the application will prompt you to enter it. ")
    parser.add_argument("--cache-initial-positions", required=False, default=False, action="store_true", help="If set to True, the program will try to reuse the existing information about players initial positions \
                        so that if there are players around the center, the user doesn't need to input to which team does the player belong")
    parser.add_argument("--cache-resolving", required=False, default=False,
                        action="store_true", help="Whether to cache user resolving ids throughout the match")
    parser.add_argument("--save-video", required=False, default=False, action="store_true",
                        help="Whether to save detections video and original video with detections")
    args = parser.parse_args()

    pitch = Pitch.load_pitch(
        args.pitch_path, args.pitch_length, args.pitch_width)
    # Setup view
    view_ = view.View(view.ViewMode.NORMAL)
    # view_.full_screen_on_monitor(constants.VIDEO_WINDOW)
    ref_img = args.workdir + "/ref_img.jpg"
    if not args.cache_homography:
        args.cache_initial_positions = False
        args.cache_resolving = False

    play_analysis(view_, pitch, args.pitch_path, args.video_path, ref_img, args.detections_path,
                  args.cache_homography, args.cache_initial_positions, args.cache_resolving, args.save_video)
    # Stop the prompting thread
    prompter.running = False
    prompter.join()
