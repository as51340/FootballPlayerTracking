from typing import Tuple, Deque
from collections import deque
import time

import cv2 as cv

import globals
import utils
import view as view_lib
import constants


class LastNFramesHelper:

    def __init__(self, n: int, view: view_lib.View) -> None:
        self.n = n
        self.storage: Deque = deque(maxlen=n)
        self.view = view

    def visualize(self, window: str, new_id: int, new_det: Tuple[int, int]):
        globals.stop_thread = False
        while not globals.stop_thread:
            time.sleep(3)
            for img in self.storage:
                img_copy = img.copy()  # copy of the image so we don't have multiple black ids
                self.view.draw_2d_obj(img_copy, str(
                    new_id), new_det, constants.BLACK, False, view_lib.DrawMode.ID)
                if globals.stop_thread:
                    break
                cv.imshow(window, img_copy)
                k = cv.waitKey(1) & 0xFF
                utils.check_kill(k)
                if k == ord('p'):
                    utils.pause()
                time.sleep(0.02)
