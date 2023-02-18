from collections import deque
import time

import cv2 as cv

import globals
import utils


class LastNFramesHelper:
    
    def __init__(self, n: int) -> None:
        self.n = n
        self.storage = deque(maxlen=n)
        
    def visualize(self, window: str):
        globals.stop_thread = False
        while not globals.stop_thread:
            time.sleep(3)
            for img in self.storage:
                if globals.stop_thread:
                    break
                cv.imshow(window, img)
                k = cv.waitKey(1) & 0xFF
                utils.check_kill(k)
                if k == ord('p'):
                    utils.pause()
                time.sleep(0.02)