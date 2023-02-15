from typing import Tuple, List

from team import Team
from pitch import Pitch
from person import Player, Referee, Person
import constants

class Match:
    """Represents the current state of the match.
    """
    def __init__(self, referee: Referee, team1_name: str, team2_name: str) -> None:
        # TODO: change this so that is receives whole referee
        self.referee = referee
        self.team1 = Team(team1_name, constants.BLUE)
        self.team2 = Team(team2_name, constants.RED)
        self.max_id = -1
        self.ignore_ids = []  # IDS that are ignored. They get added into this data structure from the user's side, e.g. assistant referee or any other object that user doesn't want to track anymore.
       
    def find_person_with_id(self, id: int) -> Person:
        """Checks whether the person with id exists in the Match structure.

        Args:
            id (int): id of the player

        Returns:
            Player: None if the player doesn't exist, else a reference to the existing player.
        """
        if id in self.referee.ids:
            return self.referee
        return self.find_player_with_id(id)
    
    def find_player_with_id(self, id: int) -> Player:
        """Checks whether the player with id exists in the Match structure.

        Args:
            id (int): player id

        Returns:
            Player: reference to the player or None
        """
        p1 = self.team1.get_player(id)
        if p1 is not None:
            return p1
        return self.team2.get_player(id)

    @classmethod
    def initialize_match(cls, pitch: Pitch, frame_detections: Tuple[int, int], bb_info: Tuple[int, int, int, int], object_ids: List[int], referee_id: int, team1_name: str, team2_name: str) -> Tuple["Match", List[Tuple[int, Tuple[int, int]]]]:
        """Initializes match with a referee and two teams. If the pitch is oriented horizontally, the first team is considered the one with the players on the left.
        If the pitch is vertically displayed, the first team is considered the one with the players above.

        Args:
            pitch (Pitch): a reference to the pitch instance
            frame_detections (Tuple[int, int]): object detections in the first frame
            object_ids (List[int]): ids of the object from the first frame
            referee_id (int): referee id

        Returns:
            Tuple[Match, List[int]]: Instance of the match and all objects that couldn't be initially classified.
        """
        match = Match(Referee("Pierluigi Collina", referee_id, constants.YELLOW), team1_name, team2_name)
        uncertain_objs = []  # objects near the center
        for i, frame_detection in enumerate(frame_detections):
            if object_ids[i] == referee_id:
                continue
            init_class_res = pitch.get_team_by_position(frame_detection)
            if init_class_res == 1:
                match.team1.players.append(Player("name", -1, object_ids[i]))
            elif init_class_res == 2:
                match.team2.players.append(Player("name", -1, object_ids[i]))
            else:
                uncertain_objs.append((object_ids[i], bb_info[i]))
            match.max_id = max(match.max_id, object_ids[i])

        return match, uncertain_objs

