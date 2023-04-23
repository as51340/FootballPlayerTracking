from collections import deque
from typing import Tuple

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

