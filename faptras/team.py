from typing import List, Tuple

class Team:
    
    def __init__(self, color: Tuple[int, int, int]) -> None:
        self.players = set()
        self.color = color