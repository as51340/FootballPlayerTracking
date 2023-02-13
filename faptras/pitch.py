from __future__ import annotations
from typing import Tuple
from enum import Enum
from ast import literal_eval

import config

PitchOrientation = Enum('PitchOrientation', ['LENGTH', 'WIDTH'])


class Pitch:
    
    def __init__(self, img_path, upper_left_corner: Tuple[int, int], down_left_corner: Tuple[int, int], down_right_corner: Tuple[int, int], upper_right_corner: Tuple[int, int]) -> None:
        self.img_path = img_path # don't save image because we need multiple copies of this img
        self.upper_left_corner = upper_left_corner
        self.down_left_corner = down_left_corner
        self.down_right_corner = down_right_corner
        self.upper_right_corner = upper_right_corner
        # No matter orientation of the artifical pitch, openCv keeps the same orientation
        x_value = upper_right_corner[0] - upper_left_corner[0]
        y_value = down_left_corner[1] - upper_left_corner[1]
        if x_value > y_value:
            self.x_dim = PitchOrientation.LENGTH
            self.length, self.width = x_value, y_value
        else:
            self.x_dim = PitchOrientation.WIDTH
            self.length, self.width = y_value, x_value
    
    def __repr__(self) -> str:
        s = f"UP_LEFT: {self.upper_left_corner} DOWN_LEFT: {self.down_left_corner} DOWN_RIGHT: {self.down_right_corner} UPPER_RIGHT: {self.upper_right_corner}\n"
        s += f"length: {self.length} width: {self.width}\n"
        s += f"orientation: {self.x_dim}"
        return s

    def __str__(self) -> str:
        return self.__repr__()
    
    @classmethod
    def load_pitch(cls, pitch_path: str) -> Pitch:
        # Get the name of the file only without extension
        last_slash_ind, last_dot_ind = pitch_path.rfind('/'), pitch_path.rfind('.')
        pitch_name = pitch_path[last_slash_ind+1:last_dot_ind]
        with open(config.path_to_pitches_config, "r") as pitches_config_file:
            while (line := pitches_config_file.readline().rstrip()):
                if line.startswith('#'):
                    continue
                pitch_data = line.split(" ")
                if pitch_data[0] == pitch_name:
                    return Pitch(pitch_path, *(map(lambda tup_str: literal_eval(tup_str), pitch_data[1:])))
        return None
    
    def get_team_by_position(self, frame_detection: Tuple[int, int]):
        """Returns initial team by the coord of the player. Returns True for the first part of the pitch and False for the second part of the pitch. When the pitch has horizontal layoff, the first part
        is the left part and when the pitch has the horizontal layoff, the first part is considered the upper part of the pitch.

        Args:
            frame_detection: Tuple[int, int]
        """
        if (self.x_dim == PitchOrientation.LENGTH and frame_detection[0] < self.length / 2) or \
            (self.x_dim == PitchOrientation.WIDTH and  frame_detection[1] < self.length / 2 ):
                return True
        return False
    

if __name__ == "__main__":
    pitch = Pitch.load_pitch("./pitches_data/green_pitch_rotated_1/green_pitch_rotated_1.jpg")
    print(pitch)