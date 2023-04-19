from enum import Enum

import cv2 as cv

SituationMode = Enum("SituationMode", ["SAVE", "NO_SAVE"])

class GameSituations:
    
    def __init__(self, width, height, fps_rate) -> None:
        self.situation_mode = SituationMode.NO_SAVE
        fourcc = cv.VideoWriter_fourcc(*'mp4v') 
        self.video = cv.VideoWriter('game_situations.mp4', fourcc, fps_rate, (width, height))
        self.needs_release = False

    def needs_saving(self):
        if self.situation_mode == SituationMode.SAVE:
            self.needs_release = True
            return True
        return False
    
    def switch_mode(self):
        if self.situation_mode == SituationMode.NO_SAVE:
            self.situation_mode = SituationMode.SAVE
        else:
            self.situation_mode = SituationMode.NO_SAVE
    