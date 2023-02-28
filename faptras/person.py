from typing import List, Tuple
import math

class Person:
    
    def __init__(self, name: str, initial_id: int, initial_position: Tuple[int, int]) -> None:
        """Referees and players will have initial_id. This is different from player's jersey number. All persons can have multiple ids -> those are ids given by the detection algorithm.

        Args:
            name (str): Person's name in the real life.
            initial_id (int): Initial id of this detection.
            initial_position (Tuple[int, int]): Player's initial position in 2D space in pixels.
        """
        self.name = name
        self.ids: List[int] = [initial_id]
        self.current_position: Tuple[int, int] = int(initial_position[0]), int(initial_position[1])
        self.total_run = 0  # How much player run estimated in meters
    

    def update_total_run(self, new_position: Tuple[int, int]) -> None:
        """Updates the amount that player run. The distance is calculated as Euclidean distance between last two person's positions.
        Current position and new position are in meters units.
        z = math.sqrt((x1-x2)**2 + (y1 - y2)**2)

        Args:
            new_position (Tuple[int, int]): New person's position
        """
        distance_run =  math.sqrt((self.current_position[0] - new_position[0])**2 + (self.current_position[1] - new_position[1])**2)
        # print(f"Distance run: {distance_run}")
        self.total_run += distance_run
        self.current_position = new_position
    
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
