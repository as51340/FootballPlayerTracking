from typing import List, Tuple, Deque, Dict
from collections import deque, defaultdict

import utils
import constants
import pitch

class Person:
    
    def __init__(self, name: str, initial_id: int, initial_position: Tuple[float, float]) -> None:
        """Referees and players will have initial_id. This is different from player's jersey number. All persons can have multiple ids -> those are ids given by the detection algorithm.

        Args:
            name (str): Person's name in the real life.
            initial_id (int): Initial id of this detection.
            initial_position (Tuple[int, int]): Player's initial position in 2D space in pixels.
        """
        self.name = name
        self.ids: List[int] = [initial_id]
        self.current_position: Tuple[float, float] = initial_position  # in 2D space in pixels
        self.sum_pos_x, self.sum_pos_y = initial_position[0], initial_position[1]
        self.total_times_seen = 1
        self.total_run = 0  # How much player run estimated in meters
        self.last_seen_frame_id = 1
        self.speeds: Deque[float] = deque()  # needed just as cache
        self.sprint_categories: Dict[utils.SprintCategory, List[float]] = defaultdict(list)
        self.current_sprint_category = None
         # stores all positions in absolute pixels but not every frame, it is sampled every N-th frame
         # to calibrate for the camera imprecision
        self.positions: Dict[int, Tuple[float, float]] = {1: initial_position} 
        self.all_positions: Dict[int, Tuple[float, float]] = {1: initial_position} 

    def update_total_run(self, pitch: pitch.Pitch, new_position: Tuple[float, float], current_frame: int) -> None:
        """Updates the amount that player run. The distance is calculated as Euclidean distance between last two person's positions.
        Current position and new position are in meters units.
        z = math.sqrt((x1-x2)**2 + (y1 - y2)**2)

        Args:
            new_position (Tuple[float, float]): New person's position in pixels.
        """
        # Distance update
        new_position_pixels = new_position
        new_position = pitch.pixel_to_meters_positions(new_position)
        current_position = pitch.pixel_to_meters_positions(self.current_position)
        distance_run = utils.calculate_euclidean_distance(current_position, new_position)
        self.total_run += distance_run
        self.sum_pos_x += new_position[0]
        self.sum_pos_y += new_position[1]
        # Speed update
        vx_local = new_position[0] - current_position[0]
        vy_local = new_position[1] - current_position[1]
        v = utils.calculate_speed_magnitude(vx_local, vy_local)
        self.speeds.append(v)
        if len(self.speeds) == constants.SPEED_AVG_NUM:
            speed_avg = sum(self.speeds) / constants.SPEED_AVG_NUM
            sprint_category = utils.get_sprint_category(speed_avg)
            self.speeds.popleft()
            if self.current_sprint_category is None:
                self.current_sprint_category = sprint_category
                self.start_sprint_position = new_position
            elif sprint_category < self.current_sprint_category:
                sprint_distance = utils.calculate_euclidean_distance(self.start_sprint_position, new_position)
                self.sprint_categories[self.current_sprint_category].append(sprint_distance)
                self.start_sprint_position = new_position
                self.current_sprint_category = sprint_category
            elif sprint_category > self.current_sprint_category:
                self.current_sprint_category = sprint_category
                
        self.total_times_seen += 1
        self.current_position = new_position_pixels
        self.positions[current_frame] = new_position_pixels
    
    @property
    def label(self) -> str:
        return ""

class Referee(Person):
    
    def __init__(self, name: str, initial_id: int, color: Tuple[int, int, int, int], initial_position: Tuple[int, int]) -> None:
        """Initializes referee. By taking into the account coloring, referee is on the same level as Team.
        Args:
            name (str): Person's name in the real life.
            initial_id (int): Initial id of this detection
            color (Tuple[int, int, int]): Color used to draw label on video and circles on detections window.
            initial_position (Tuple[int, int]): Player's initial position in 2D space in pixels.
        """
        super().__init__(name, initial_id, initial_position)
        self.color = color


class Player(Person):
    
    def __init__(self, name: str, jersey_number: int, initial_id: int, initial_position: Tuple[int, int]) -> None:
        """Initializes player.

        Args:
            name (str): Name of the player.
            jersey_number (int): Player's jersey number.
            initial_id (int): Initial id of the player from the tracking system.
            initial_position (Tuple[int, int]): Player's initial position in 2D space in pixels.
        """
        super().__init__(name, initial_id, initial_position)
        self.jersey_number = jersey_number
    
    @property
    def label(self) -> str:
        return self.ids[0]
        # return str(self.jersey_number)
