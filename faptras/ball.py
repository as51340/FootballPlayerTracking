from collections import deque
from typing import Tuple, List

import constants
from smoother import SmootherPosition

import numpy as np


class Ball(SmootherPosition):

    def __init__(self) -> None:
        super().__init__()
        self.current_position = None
        self.color = constants.WHITE
        self.positions = []  # All positions of the ball

    def update_position(self, new_position: Tuple[float, float]) -> None:
        """Updates ball position."""
        self.current_position = new_position
        self.positions.append(new_position)

    def get_last_n_positions(self, frames: int, window: int) -> List[np.array]:
        """Return last N positions of the ball. Doesn't take into account if the ball wasn't tracked in some frames. The caller should take care about the size of the returning window."""
        ma_window = np.ones(window) / window
        ball_positions = []
        ball_positions = np.array(self.positions[-frames:])
        ball_positions = np.apply_along_axis(lambda dim: np.convolve(
            dim, ma_window, mode="valid"), axis=0, arr=ball_positions)
        print(f"Ball positions: {ball_positions.shape}")
        return ball_positions
