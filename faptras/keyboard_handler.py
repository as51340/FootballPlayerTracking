import time

import cv2 as cv

import view
import utils
import constants
import analytics_viewer
import pitch
import match
from game_situations import GameSituations


def restart_visualizations():
    time.sleep(2)
    cv.namedWindow(constants.DETECTIONS_WINDOW)
    view.View.full_screen_on_monitor(constants.VIDEO_WINDOW)


def forward_analytics_call(function, *args):
    cv.destroyAllWindows()
    function(*args)
    restart_visualizations()


def forward_analytics_calls(functions, *args):
    cv.destroyAllWindows()
    for function in functions:
        function(*args)
    restart_visualizations()


def handle_key_press(k, view: view.View, analytics_display: analytics_viewer.AnalyticsViewer, pitch: pitch.Pitch,
                     match: match.Match, fps_rate: int, frame_id: int, game_situations: GameSituations):
    """Handles key press at the end of each frame. Returns true if the visualization needs to be ended. False otherwise."""
    utils.check_kill(k)
    seek_frames = 0
    if k == ord('q'):
        return True, seek_frames  # Quit visualization
    elif k == ord('b'):  # match acc summary
        forward_analytics_call(analytics_display.show_match_acc_summary,
                               pitch, match, fps_rate, constants.SMOOTHING_AVG_WINDOW)
    elif k == ord('c'):  # convex hull for a team
        forward_analytics_call(
            analytics_display.draw_convex_hull_for_players, match, pitch, frame_id)
    elif k == ord('d'):  # delaunay tessellation
        forward_analytics_call(
            analytics_display.draw_delaunay_tessellation, match, pitch, frame_id)
    elif k == ord('e'):  # dynamic pitch control
        forward_analytics_call(analytics_display.dynamic_pitch_control, pitch, match, min(
            120, int(frame_id / fps_rate)), fps_rate, constants.POSITION_SMOOTHING_AVG_WINDOW)
    elif k == ord('f'):  # Switch full screen mode
        view.switch_screen_mode()  # Switch between normal and full mode
    elif k == ord('h'):  # heat map for each player
        print(f"Please enter player id: ")
        try:
            forward_analytics_call(
                analytics_display.draw_player_heatmap, match, pitch, int(input()))
        except ValueError:
            print(f"Wrong input, please restart your calculations...")
            time.sleep(2)
    elif k == ord('j'):  # rewind back the video
        seek_frames = -5 * fps_rate
    elif k == ord('l'):  # rewind forward the video
        seek_frames = 5 * fps_rate
    elif k == ord('m'):  # whether to save highlights of the match or not
        game_situations.switch_mode()
    elif k == ord('p'):  # pause visualization
        utils.pause()  # Pause visualization
    elif k == ord('r'):  # total run
        forward_analytics_call(analytics_display.show_match_total_run,
                               pitch, match, fps_rate, constants.SMOOTHING_AVG_WINDOW)
    elif k == ord('s'):  # switch view mode
        view.switch_draw_mode()  # Switch drawing mode between circles and ids
    elif k == ord('t'):  # sprints
        forward_analytics_call(analytics_display.show_match_sprint_summary,
                               pitch, match, fps_rate, constants.SMOOTHING_AVG_WINDOW)
    elif k == ord('u'):  # dynamic voronoi diagrams
        forward_analytics_call(
            analytics_display.draw_dynamic_voronoi_diagrams, match, pitch, min(
                10, int(frame_id / fps_rate)), constants.POSITION_SMOOTHING_AVG_WINDOW)
    elif k == ord('v'):  # voronoi diagrams
        forward_analytics_call(
            analytics_display.draw_voronoi_diagrams, match, pitch, frame_id)
    return False, seek_frames
