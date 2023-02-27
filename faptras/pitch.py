from __future__ import annotations
from typing import Tuple, List
from enum import Enum
from ast import literal_eval

import config
import utils

PitchOrientation = Enum('PitchOrientation', ['LENGTH', 'WIDTH'])


class Pitch:
    
    def __init__(self, img_path, upper_left_corner: Tuple[int, int], down_left_corner: Tuple[int, int], down_right_corner: Tuple[int, int], 
                 upper_right_corner: Tuple[int, int], pitch_length: int, pitch_width: int) -> None:
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
            self.length, self.width = x_value, y_value  # pixels
        else:
            self.x_dim = PitchOrientation.WIDTH
            self.length, self.width = y_value, x_value  # pixels
        self.pitch_length = pitch_length  # in meters
        self.pitch_width = pitch_width  # in meters
    
    def __repr__(self) -> str:
        s = f"UP_LEFT: {self.upper_left_corner} DOWN_LEFT: {self.down_left_corner} DOWN_RIGHT: {self.down_right_corner} UPPER_RIGHT: {self.upper_right_corner}\n"
        s += f"length: {self.length} width: {self.width}\n"
        s += f"orientation: {self.x_dim}"
        return s

    def __str__(self) -> str:
        return self.__repr__()
    
    def pixel_to_meters_positions(self, pixel_coords: Tuple[int, int]) -> Tuple[float, float]:
        """Computes object's position in real-world (meter) coordinates from pixel positions/

        Args:
            pixel_coords (Tuple[int, int]): Object coordinates in pixels.

        Returns:
            Tuple[float, float]: Object coordinate in meters.
        """
        if self.x_dim == PitchOrientation.LENGTH:
            return pixel_coords[0] * self.pitch_length / self.length, pixel_coords[1] * self.pitch_width / self.width
        return pixel_coords[0] * self.pitch_width / self.width, pixel_coords[1] * self.pitch_length / self.length
    
    @classmethod
    def load_pitch(cls, pitch_path: str, pitch_length: int, pitch_width: int) -> Pitch:
        """Creates pitch from given information.

        Args:
            pitch_path (str): Path to the pitch info file.
            pitch_length (int): Pitch length in meters
            pitch_width (int): Pitch width in meters.

        Returns:
            Pitch: Loaded pitch.
        """
        # Get the name of the file only without extension
        pitch_name = utils.get_file_name(pitch_path)
        with open(config.PATH_TO_PITCHES, "r") as pitches_config_file:
            while (line := pitches_config_file.readline().rstrip()):
                if line.startswith('#'):
                    continue
                pitch_data = line.split(" ")
                if pitch_data[0] == pitch_name:
                    return Pitch(pitch_path, *(map(lambda tup_str: literal_eval(tup_str), pitch_data[1:])), pitch_length, pitch_width)
        return None
    
    def get_team_by_position(self, frame_detection: Tuple[int, int]) -> int:
        """Returns initial team by the coord of the player. Returns 1 for the first part of the pitch and 2 for the second part of the pitch. When the pitch has horizontal layoff, the first part
        is the left part and when the pitch has the horizontal layoff, the first part is considered the upper part of the pitch. Returns 2 for uncertain players near the centre.

        Args:
            frame_detection: Tuple[int, int]
        """
        if self.x_dim == PitchOrientation.LENGTH:
            if frame_detection[0] < self.upper_left_corner[0] + self.length / 2 - 10:
                return 1  # certain that is the first team
            elif frame_detection[0] > self.upper_left_corner[0] + self.length / 2 + 10:
                return 2
            else:
                return 0  # second team
        else:  # width orientation
            if frame_detection[1] < self.upper_left_corner[1] + self.width / 2 - 10:
                return 1  # certain that is the first team
            elif frame_detection[1] > self.upper_left_corner[1] + self.width / 2 + 10:
                return 2
            else:
                return 0  # second team            
    
    def is_detection_outside(self, detection_x_coord: int, detection_y_coord: int) -> bool:
        """Checks whether the detection is outside of the pitch.

        Args:
            pitch (Pitch): A reference to the pitch.
            detection_x_coord (int): x_coordinate in 2D pitch.
            detection_y_coord (int): y_coordinate in 2D pitch.

        Returns:
            bool: true if it is outside of the pitch.
        """
        if detection_x_coord < self.upper_left_corner[0] or detection_y_coord < self.upper_left_corner[1] or \
            detection_x_coord > self.upper_right_corner[0] or detection_y_coord > self.down_right_corner[1]:
            return True
        return False

    def get_objects_within(self, detections, bb_info, object_ids) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]], List[int]]:
        """Returns detections (2D objects), bounding boxes and object_ids only of objects which are within the pitch boundary based on the 2D image.

        Args:
            detections (_type_): 2D detections.
            bb_info (_type_): Bounding box information.
            object_ids (_type_): Objects ids

        Returns:
            2d detections, bounding boxes and ids of the objects inside the pitch.
        """
        detections_in_pitch, bb_info_ids, object_ids_in_pitch = [], [], []
        for i, detection in enumerate(detections):
            if not self.is_detection_outside(detection[0], detection[1]):
                detections_in_pitch.append(detection)
                bb_info_ids.append(bb_info[i])
                object_ids_in_pitch.append(object_ids[i])
        return detections_in_pitch, bb_info_ids, object_ids_in_pitch
    
if __name__ == "__main__":
    pitch = Pitch.load_pitch("./pitches_data/green_pitch_rotated_1/green_pitch_rotated_1.jpg")
    print(pitch)