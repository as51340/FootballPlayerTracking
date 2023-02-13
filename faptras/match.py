from typing import Tuple, List

from team import Team
from pitch import Pitch
import constants

class Match:
    """Represents the current state of the match.
    """
    def __init__(self, referee_id: int) -> None:
        self.referee_id = referee_id
        self.referee_color = constants.YELLOW
        self.team1 = Team(constants.BLUE)
        self.team2 = Team(constants.RED)
        
    
    @classmethod
    def initialize_match(cls, pitch: Pitch, frame_detections: Tuple[int, int], object_ids: List[int], referee_id: int):
        match = Match(referee_id)
        for i, frame_detection in enumerate(frame_detections):
            if object_ids[i] == referee_id:
                continue
            if pitch.get_team_by_position(frame_detection):
                match.team1.players.add(object_ids[i])
            else:
                match.team2.players.add(object_ids[i])
        return match

