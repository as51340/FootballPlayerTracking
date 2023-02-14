# class used for handling view, we will use it for refactoring 
from enum import Enum
from typing import Tuple

import cv2 as cv
import numpy as np

import constants
from monitor_utils import get_offset_to_second_monitor

ViewMode = Enum('ViewMode', ['FULL', 'NORMAL'])  # full screen vs normal mode

TEXT_FACE = cv.FONT_HERSHEY_DUPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2

class View:
    
    def __init__(self, view_mode: ViewMode) -> None:
        self.last_clicked_windows = []
        self.view_mode = view_mode
    
    @classmethod
    def full_screen_on_monitor(cls, window_name):
        cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
        monitor_info = get_offset_to_second_monitor()
        cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.moveWindow(window_name, monitor_info[0], monitor_info[1]);    

    @classmethod
    def draw_person(cls, frame_img: np.ndarray, text: str, center: Tuple[int, int], text_color: Tuple[int, int, int]):
        """Draws person as a circle and a text inside.
        """
        # cv.circle(frame_img, center, radius, circle_color, -1)    
        text_size, _ = cv.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2))
        cv.putText(frame_img, text, text_origin, TEXT_FACE, TEXT_SCALE, text_color, TEXT_THICKNESS, cv.LINE_AA)
        
    
    @classmethod
    def box_label(cls, frame: np.ndarray, bb_info: Tuple[int, int, int, int], rec_color,  label='', txt_color=constants.WHITE):
        """Draws label on a video frame. 

        Args:
            frame (np.ndarray): video frame
            bb_info (Tuple[int, int, int, it]): bounding box detection format in MOT format.
            rec_color (_type_): color for drawing rectangle.
            label (str, optional): Label to draw on the player. Defaults to ''.
            txt_color (_type_, optional): color of the text. Defaults to constants.WHITE.
        """
        lw = 2
        label = str(label)
        # Extract from MOT format
        upper_left_corner = (bb_info[0], bb_info[1])
        down_right_corner = (bb_info[0] + bb_info[2], bb_info[1] + bb_info[3])
        # Draw label
        cv.rectangle(frame, upper_left_corner, down_right_corner, rec_color, thickness=lw, lineType=cv.LINE_AA)
        
        tf = 1  # font thickness
        w, h = cv.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        
        outside = upper_left_corner[1] - h >= 3
        down_right_corner = upper_left_corner[0] + w, upper_left_corner[1] - h - 3 if outside else upper_left_corner[1] + h + 3
        cv.rectangle(frame, upper_left_corner, down_right_corner, rec_color, -1, cv.LINE_AA)  # filled
        cv.putText(frame,
                    label, (upper_left_corner[0], upper_left_corner[1] - 2 if outside else upper_left_corner[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv.LINE_AA)

    @classmethod
    def draw_old_circles(cls, img, points):
        """Draws all saved points on an image. Used for implementing undo buffer.
        Args:
            img (np.ndarray): A reference to the image.
            points (List[List[int]]): Points storage.
        """
        for x, y in points:
            cv.circle(img, (x,y), 5, constants.RED, -1)
    
    @classmethod
    def read_image(cls, img_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Reads image from the given path and creates its copy.
        Args:
            img_path (str): Path of the image.
        """
        # Registers a mouse callback function on a image's copy.
        img = cv.imread(img_path, -1)
        img_copy = img.copy()
        return img, img_copy

    def switch_screen_mode(self):
        """Switches screen mode
        """
        if self.view_mode == ViewMode.FULL:
            print("Changing to resized")
            cv.setWindowProperty(constants.VIDEO_WINDOW, cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
            self.view_mode = ViewMode.NORMAL
        else:
            print("Changing to full")
            View.full_screen_on_monitor(constants.VIDEO_WINDOW)
            self.view_mode = ViewMode.FULL

    # mouse callback function
    def _select_points_wrapper(self, event, x, y, _, params):
        """Wrapper for mouse callback.
        """
        global last_clicked_window
        window, points, img_copy = params
        if event == cv.EVENT_LBUTTONDOWN:
            self.last_clicked_windows.append(window)
            cv.circle(img_copy, (x,y), 5, constants.RED, -1)
            points.append([x, y])
