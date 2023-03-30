import time

import cv2 as cv

import view
import utils
import constants
import analytics_viewer
import pitch
import match


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

def handle_key_press(k, view: view.View, analytics_display: analytics_viewer.AnalyticsViewer, pitch: pitch.Pitch, match: match.Match, fps_rate: int, frame_id: int):
    """Handles key press at the end of each frame. Returns true if the visualization needs to be ended. False otherwise."""
    utils.check_kill(k)
    if k == ord('q'):
        return True  # Quit visualization
    if k == ord('f'):
        view.switch_screen_mode() # Switch between normal and full mode
    elif k == ord('s'):
        view.switch_draw_mode() # Switch drawing mode between circles and ids
    elif k == ord('p'):
        utils.pause() # Pause visualization
    elif k == ord('r'):  # total run
        forward_analytics_call(analytics_display.show_match_total_run, pitch, match, fps_rate, 7)
    elif k == ord('t'):  # sprints
        forward_analytics_call(analytics_display.show_match_sprint_summary, pitch, match, fps_rate, 7)
    elif k == ord('h'):  # heat map for each player
        print(f"Please enter player id: ")
        try:
            forward_analytics_call(analytics_display.draw_player_heatmap, match, pitch, int(input()))
        except ValueError:
            print(f"Wrong input, please restart your calculations...")
            time.sleep(2)
    elif k == ord('c'):  # convex hull for a team
        print("Please enter team's name: ")
        team_name = input()
        team, left = None, True  # on which side is the team's goalkeeper
        if team_name == match.team1.name:
            team = match.team1
        elif team_name == match.team2.name:
            team = match.team2
            left = False
        else:
            print(f"Unknown team, please start calculations again...")
            time.sleep(2)
        if team is not None:
            forward_analytics_call(analytics_display.draw_convex_hull_for_players, pitch, team, frame_id, left)
    elif k == ord('d'):  # delaunay tessellation
        forward_analytics_call(analytics_display.draw_delaunay_tessellation, match, pitch, frame_id)
    elif k == ord('v'):  # voronoi diagrams
        forward_analytics_call(analytics_display.draw_voronoi_diagrams, match, pitch, frame_id)
    elif k == ord('a'):  # animations
        forward_analytics_call(analytics_display.visualize_animation, match, pitch, min(5, int(frame_id / fps_rate)), frame_id)
    return False

    