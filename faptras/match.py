from typing import Tuple, List, Dict

import numpy as np

from team import Team
from pitch import Pitch
from person import Player, Referee, Person
import constants
import view
import resolve_helpers
import utils

class Match:
    """Represents the current state of the match.
    """
    def __init__(self, team1_name: str, team2_name: str) -> None:
        self.team1 = Team(team1_name, constants.BLUE)
        self.team2 = Team(team2_name, constants.RED)
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
    
    def resolve_team_helper(self, id: int, detection_info: Tuple[float, float], team: int, jersey_number: int, name: str):
        """Adds player to the team based on the team name and returns True. If no such team exists in the match, the method returns false.

        Args:
            team (str): Team name`
            id (int): Player id which needs to be added to some team.
            detection_info (Tuple[int, int]): Initial player position in 2D space in meters.
            jersey_number (int): Jersey number of the player. If it is referee, then it will be ignored.

        Returns:
            bool: True if team matches one of the teams' names, False otherwise.
        """
        if team == 0:
            self.team1.players.append(Player(name, jersey_number, id, detection_info))
            return True
        elif team == 1:
            self.team2.players.append(Player(name, jersey_number, id, detection_info))
            return True
        elif team == 2:
            self.referee = Referee(name, id, constants.YELLOW, detection_info)
            return True
        return False

    def get_new_objects(self, bb_infos: Tuple[int, int, int, int], object_ids: List[int], detections_in_pitch: List[Tuple[int, int]]) -> List[int]:
        """Returns new objects for the current frame based on the information on ignoring ids and teams.

        Args:
            bb_infos (Tuple[int, int, int, int]): Bounding boxes. 
            object_ids (List[int]): List of object identifiers.

        Returns:
            List[Tuple[int, int, int, int], int]: New objects' bounding boxes and identifiers.
        """
        new_objects: List[Tuple[int, int, int, int], int] = []
        for i, obj_id in enumerate(object_ids):
            if obj_id not in self.ignore_ids and self.find_person_with_id(obj_id) is None:
                new_objects.append((detections_in_pitch[i], bb_infos[i], obj_id))
        return new_objects
    
    def resolve_user_action(self, action: int, obj_id: int, det: Tuple[int, int], resolving_positions_cache: Dict[int, int], resolve_helper: resolve_helpers.LastNFramesHelper, prompter):
        """Resolves user input based on a few simple conditions.

        Args:
            action (int): Action that user requested.
            obj_id (int): Id of the object that needs resolving
            resolve_helper (resolve_helpers.LastNFramesHelper): Instance of the helper.
        """
        resolving_positions_cache[obj_id] = action  # if recursively call, it will be overwritten
        if action == 0:
            self.referee.ids.append(obj_id)
        elif action == -1:
            print(f"Ignoring id: {obj_id}")
            self.ignore_ids.append(obj_id)  # from now on ignore this id
        elif action == -2:
            prompter.set_execution_config(constants.prompt_input)
            resolve_helper.visualize(constants.DETECTIONS_WINDOW, obj_id, det)
            new_action = int(prompter.value)
            self.resolve_user_action(new_action, obj_id, det, resolving_positions_cache, resolve_helper, prompter)
        else:
            ex_player = self.find_player_with_id(action)
            if ex_player is not None:
                print(f"Found ex player with ids: {ex_player.ids}")
                ex_player.ids.append(obj_id)
            else:
                print(f"Ignoring id: {obj_id}")

    @classmethod
    def cache_team_resolution(cls, pitch: Pitch, cache_file: str):
        """Resolved initial informations about players and the referee from the cached file.
        Args:
            pitch (Pitch): A reference to the pitch.
            cache_file (str): Path to the file where all information is being saved.

        Returns:
            None if the process went wrong, match otherwise.
        """
        players_data: Tuple[int, Tuple[str, Tuple[int, int]]] = dict()
        try:    
            with open(cache_file, "r") as cache_file:
                team1_name = cache_file.readline().strip().rstrip()
                team2_name = cache_file.readline().strip().rstrip()
                while (line := cache_file.readline().rstrip()):
                    line_data = line.split('###')
                    players_data[int(line_data[0])] = (line_data[1], (line_data[2], line_data[3])) 
        except Exception:
            return False

        match = Match(team1_name, team2_name)
        for obj_id, (line, initial_position) in players_data.items():
            values = list(map(lambda val: val.strip().rstrip(), line.split(',')))
            team = int(values[0])
            jersey_number = int(values[1])
            if not match.resolve_team_helper(obj_id, pitch.pixel_to_meters_positions(utils.to_tuple_int(initial_position)), team, jersey_number, values[2]):
                return None
        return match
    
    @classmethod
    def user_team_resolution(cls, pitch: Pitch, obj_ids: List[int], detections_info: List[Tuple[int, int]], bb_info: List[Tuple[int, int, int, int]], img: np.ndarray, window: str, cache_file: str, prompter) -> None:
        """User manually decides about the team of the player shown on the image. If user inputs the team that is not part of the current match, the procedure is repeated for the same player.

        Args:
            pitch (Pitch): _description_
            obj_ids (List[int]): _description_
            detections_info (List[Tuple[int, int]]): _description_
            bb_info (List[Tuple[int, int, int, int]]): _description_
            img (np.ndarray): _description_
            window (str): _description_
            cache_file (str): _description_
            prompter (_type_): _description_

        Returns:
            _type_: _description_
        """
        print("Please insert the name of the first team: ")
        team_name1 = input()
        print("Please insert the name of the second team: ")
        team_name2 = input()
        match = Match(team_name1, team_name2)
        player_cache: Tuple[int, Tuple[str, Tuple[int, int]]] = dict()
        for i, bb_obj_info in enumerate(bb_info):
            show_frame = img.copy() 
            view.View.box_label(show_frame, bb_obj_info, constants.BLACK, obj_ids[i])
            while True:
                prompter.set_execution_config("Insert 0 for the first team, 1 for the second and 2 for the referee, jersey_number or -1, name: ")
                view.View.show_img_while_not_killed(window, show_frame)
                value = prompter.value
                values = prompter.value.split(',')
                values = list(map(lambda val: val.strip().rstrip(), values))
                team = int(values[0])
                jersey_number = int(values[1])
                try:
                    if not match.resolve_team_helper(obj_ids[i], pitch.pixel_to_meters_positions(detections_info[i]), team, jersey_number, values[2]):
                        print("Unknown team, please insert again...")
                    else:
                        player_cache[obj_ids[i]] = value, detections_info[i]
                        break
                except ValueError:
                    print("Wrong format, try again")

        with open(cache_file, "w") as file:
            file.write(f"{team_name1}\n")
            file.write(f"{team_name2}\n")
            for obj_id, (line, initial_position) in player_cache.items():
                print(f"Initial position: {detections_info}")
                file.write(f"{obj_id}###{line}###{initial_position[0]}###{initial_position[1]}\n")
        return match

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
        match = Match(Referee("Pierluigi Collina", referee_id, constants.YELLOW) , team1_name, team2_name)
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

