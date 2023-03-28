from typing import List, Tuple

from person import Player

class Team:
    
    def __init__(self, team_name: str, color: Tuple[int, int, int]) -> bool:
        self.name = team_name
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

    def get_player_positions_from_frame(self, frame: int):
        """Retrieves all players from the frame "frame". """
        return [player.all_positions[frame] for player in self.players if frame in player.all_positions.keys()]
    
    