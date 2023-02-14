# Constants
WHITE = (255, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)
# SRC_WINDOW and DST_WINDOW are used for homography calculation
# DETECTIONS_WINDOW is used for birds-eye view on the pitch
# VIDEO_WINDOW is used for playing video with detections
SRC_WINDOW, DST_WINDOW, DETECTIONS_WINDOW = "src", "dst", "det"
VIDEO_WINDOW = "vid"
prompt_input = "Do you recognize object on the image? If new object shows anything except referee and players, enter -1. \
Otherwise, if player is shown on the image please enter its jersey number and if the image shows referee, enter 0. If you don't recognize \
existing player press enter, a new player will be added and you will be prompted to enter the player's team. When you are ready to enter, please press 'q'. "  # currently the user will enter detection id

