from typing import Tuple
from collections import deque
import time

import cv2 as cv

import globals
import utils
import view
import constants


class LastNFramesHelper:
    
    def __init__(self, n: int, view_: view.View) -> None:
        self.n = n
        self.storage = deque(maxlen=n)
        self.view_ = view_
        
    def visualize(self, window: str, new_id: int, new_det: Tuple[int, int]):
        globals.stop_thread = False
        while not globals.stop_thread:
            time.sleep(3)
            for img in self.storage:
                self.view_.draw_person(img, str(new_id), new_det, constants.BLACK)
                if globals.stop_thread:
                    break
                cv.imshow(window, img)
                k = cv.waitKey(1) & 0xFF
                utils.check_kill(k)
                if k == ord('p'):
                    utils.pause()
                time.sleep(0.02)