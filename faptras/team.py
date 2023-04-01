from typing import List, Tuple

import numpy as np

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

    def get_last_n_player_positions(self, frames: int, window: int) -> List[np.array]:
        """Returns "frames" last positions of all players in the team. Doesn't take into account if some player wasn't tracked in some frames.

        Args:
            frames (int): Last N frames.
            window (int): Smoothing window

        Returns:
            List[np.array]: List of players. For each player there is a numpy array representing both dimensions for last N frames.
        """
        ma_window = np.ones(window) / window
        team_positions = []
        for player in self.players:
            player_positions = np.array(list(player.all_positions.values())[-frames:])
            player_positions = np.apply_along_axis(lambda dim: np.convolve(dim, ma_window, mode="valid"), axis=0, arr=player_positions)
            team_positions.append(player_positions)
        return team_positions
    
    