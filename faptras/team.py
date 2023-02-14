from typing import List, Tuple

from person import Player

class Team:
    
    def __init__(self, color: Tuple[int, int, int]) -> bool:
        self.players: List[Player] = []
        self.color = color
        
    def get_player(self, id: int):
        """Checks if the team contains a player identified by id. Note that each player may have multiple ids.

        Args:
            id (int): id of the player.

        Returns:
            bool: whether the team contains a player identified by the id.
        """
        # TODO: optimize with functional programming
        for player in self.players:
            if id in player.ids:
                return player
        return None
    
    