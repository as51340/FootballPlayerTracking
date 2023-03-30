from typing import List, Tuple, Dict

class Person:
    
    def __init__(self, name: str, initial_id: int, initial_position: Tuple[float, float], initial_position_meters: Tuple[float, float]) -> None:
        """Referees and players will have initial_id. This is different from player's jersey number. All persons can have multiple ids -> those are ids given by the detection algorithm.

        Args:
            name (str): Person's name in the real life.
            initial_id (int): Initial id of this detection.
            initial_position (Tuple[float, float]): Player's initial position in 2D space in pixels.
            initial_position_meters (Tuple[float, float]): Player's initial position in meters. 
        """
        self.name = name
        self.ids: List[int] = [initial_id]
        self.current_position: Tuple[float, float] = initial_position  # in 2D space in pixels
        self.sum_pos_x, self.sum_pos_y = initial_position_meters  # in meters
        self.total_times_seen = 1
        self.last_seen_frame_id = 1
        self.all_positions: Dict[int, Tuple[float, float]] = {1: initial_position}  # in 2D space in pixels

    def update_person_position(self, new_position: Tuple[float, float], new_position_meters: Tuple[float, float], current_frame_id: int) -> None:
        """Updates person position.

        Args:
            new_position (Tuple[float, float]): New position in pixels.
            new_position_meters (Tuple[float, float]): New player position in meters.
            current_frame_id (int): Frame for which we are setting position. 
        """
        # Distance update
        self.sum_pos_x += new_position_meters[0]  # meters
        self.sum_pos_y += new_position_meters[1]  # meters
        self.current_position = new_position  # pixels
        self.all_positions[current_frame_id] = new_position
        self.total_times_seen += 1
        self.last_seen_frame_id = current_frame_id
    
    @property
    def label(self) -> str:
        return ""

class Referee(Person):
    
    def __init__(self, name: str, initial_id: int, color: Tuple[int, int, int, int], initial_position: Tuple[int, int], initial_position_meters: Tuple[float, float]) -> None:
        """Initializes referee. By taking into the account coloring, referee is on the same level as Team.
        Args:
            name (str): Person's name in the real life.
            initial_id (int): Initial id of this detection
            color (Tuple[int, int, int]): Color used to draw label on video and circles on detections window.
            initial_position (Tuple[int, int]): Player's initial position in 2D space in pixels.
            initial_position_meters (Tuple[float, float]): Player's initial position in meters. 
        """
        super().__init__(name, initial_id, initial_position, initial_position_meters)
        self.color = color


class Player(Person):
    
    def __init__(self, name: str, jersey_number: int, initial_id: int, initial_position: Tuple[int, int], initial_position_meters: Tuple[float, float]) -> None:
        """Initializes player.

        Args:
            name (str): Name of the player.
            jersey_number (int): Player's jersey number.
            initial_id (int): Initial id of the player from the tracking system.
            initial_position (Tuple[int, int]): Player's initial position in 2D space in pixels.
            initial_position_meters (Tuple[float, float]): Player's initial position in meters. 
        """
        super().__init__(name, initial_id, initial_position, initial_position_meters)
        self.jersey_number = jersey_number
    
    @property
    def label(self) -> str:
        return self.ids[0]