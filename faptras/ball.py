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

