# Constants
WHITE = (255, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)
RADIUS = 5
# SRC_WINDOW and DST_WINDOW are used for homography calculation
# DETECTIONS_WINDOW is used for birds-eye view on the pitch
# VIDEO_WINDOW is used for playing video with detections
SRC_WINDOW, DST_WINDOW, DETECTIONS_WINDOW = "src", "dst", "det"
VIDEO_WINDOW = "vid"
prompt_input = "Do you recognize object on the image? If new object shows anything except referee and players, enter -1. \
Otherwise, if player is shown on the image please enter its jersey number (previous tracking id) and if the image shows referee, enter 0. "  # currently the user will enter detection id
MPLSOCCER_PITCH_LENGTH = 120
MPLSOCCER_PITCH_WIDTH = 80
FPS_ANIMATIONS = 25
# Speeds
MAX_SPEED = 11.5  # in m/s
SMOOTHING_AVG_WINDOW = 7
WALKING_MAX_SPEED = 1.9444  # 7 km/h
EASY_MAX_SPEED = 3.8889  # 7-14 km/h
MODERATE_MAX_SPEED = 5.2778 # 14-20 km/h
FAST_MAX_SPEED = 6.944  # 20-25 km/h
