from typing import Tuple, List

from team import Team
from pitch import Pitch
from person import Player, Referee, Person
import constants

class Match:
    """Represents the current state of the match.
    """
    def __init__(self, referee: Referee) -> None:
        # TODO: change this so that is receives whole referee
        self.referee = referee
        self.team1 = Team(constants.BLUE)
        self.team2 = Team(constants.RED)
        self.max_id = -1
       
    def find_player_with_id(self, id: int) -> Person:
        """Checks whether the player with id exists in the Match structure.

        Args:
            id (int): id of the player

        Returns:
            Player: None if the player doesn't exist, else a reference to the existing player.
        """
        if id not in self.referee.ids:
            p1 = self.team1.get_player(id)
            if p1 is not None:
                return p1
            return self.team2.get_player(id)
        else:
            return self.referee
    
    @classmethod
    def initialize_match(cls, pitch: Pitch, frame_detections: Tuple[int, int], object_ids: List[int], referee_id: int) -> "Match":
        """Initializes match with a referee and two teams. If the pitch is oriented horizontally, the first team is considered the one with the players on the left.
        If the pitch is vertically displayed, the first team is considered the one with the players above.

        Args:
            pitch (Pitch): a reference to the pitch instance
            frame_detections (Tuple[int, int]): object detections in the first frame
            object_ids (List[int]): ids of the object from the first frame
            referee_id (int): referee id

        Returns:
            Match: Instance of the match
        """
        match = Match(Referee("Pierluigi Collina", referee_id, constants.YELLOW))
        for i, frame_detection in enumerate(frame_detections):
            if object_ids[i] == referee_id:
                continue
            if pitch.get_team_by_position(frame_detection):
                match.team1.players.append(Player("name", -1, object_ids[i]))
            else:
                match.team2.players.append(Player("name", -1, object_ids[i]))
            match.max_id = max(match.max_id, object_ids[i])
        return match

