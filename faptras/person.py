from typing import List, Tuple


class Person:
    
    def __init__(self, name: str, initial_id: int) -> None:
        """Referees and players will have initial_id. This is different from player's jersey number. All persons can have multiple ids -> those are ids given by the detection algorithm.

        Args:
            name (str): Person's name in the real life.
            initial_id (int): Initial id of this detection.
        """
        self.name = name
        self.ids: List[int] = []
        self.ids.append(initial_id)

    @property
    def label(self) -> str:
        return ""

class Referee(Person):
    
    def __init__(self, name: str, initial_id: int, color: Tuple[int, int, int]) -> None:
        """Initializes referee. By taking into the account coloring, referee is on the same level as Team.
        Args:
            name (str): Person's name in the real life.
            initial_id (int): Initial id of this detection
            color (Tuple[int, int, int]): Color used to draw label on video and circles on detections window.
        """
        super().__init__(name, initial_id)
        self.color = color


class Player(Person):
    
    def __init__(self, name: str, jersey_number: int, initial_id: int) -> None:
        """Initializes player.

        Args:
            initial_id (int): _description_
        """
        super().__init__(name, initial_id)
        self.jersey_number = jersey_number
    
    @property
    def label(self) -> str:
        return self.ids[0]
        # return str(self.jersey_number)
