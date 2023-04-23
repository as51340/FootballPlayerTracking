from typing import Tuple
from collections import deque
import abc

import numpy as np

import constants


# Behaves as an interface for all objects whose position isn't visualized in real time but is smoothed instead.
class SmootherPosition(abc.ABC):

    def __init__(self) -> None:
        self.x_window = deque()
        self.y_window = deque()

    def update_drawing_position(self, new_position: Tuple[float, float]) -> Tuple[float, float]:
        """Updates window of positions and returns smoothed position if the window is of size constants.POSITION_SMOOTHING_AVG_WINDOW. Otherwise, returns the latest position."""
        # If we don't have enough data, then just return the latest position
        if len(self.x_window) < constants.POSITION_SMOOTHING_AVG_WINDOW:
            self.x_window.append(new_position[0])
            self.y_window.append(new_position[1])
            return new_position
        positions = np.array([self.x_window, self.y_window])
        window = np.ones(constants.POSITION_SMOOTHING_AVG_WINDOW) / \
            constants.POSITION_SMOOTHING_AVG_WINDOW
        player_smoothed_positions = np.apply_along_axis(lambda dim: np.convolve(
            dim, window, mode="valid"), axis=1, arr=positions)
        self.x_window.popleft()
        self.y_window.popleft()
        self.x_window.append(new_position[0])
        self.y_window.append(new_position[1])
        return player_smoothed_positions[0], player_smoothed_positions[1]
