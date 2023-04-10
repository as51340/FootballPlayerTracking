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
        """Creates match.

        Args:
            team1_name (str): Name of the first team.
            team2_name (str): Name of the second team.
        """
        self.team1 = Team(team1_name, constants.BLUE)  # team1 is left
        self.team2 = Team(team2_name, constants.RED)  # team2 is red
        # IDS that are ignored. They get added into this data structure from the user's side, e.g. assistant referee or any other object that user doesn't want to track anymore.
        self.ignore_ids = []
        # initial ids in the match. Needed for automatic AI resolver.
        self.initial_ids = set()

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

    def get_info_for_drawing(self, input_id: int):
        """Returns info needed for drawing based on the given id.

        Args:
            input_id (int): Id of the object that needs to be drawn.

        Returns:
            Tuple[Person, Color, str]: Reference to person, color and its id to be shown.
        """
        team1_player = self.team1.get_player(input_id)
        team2_player = self.team2.get_player(input_id)
        if input_id in self.referee.ids:
            return self.referee, self.referee.color, str(input_id)
        elif team1_player is not None:
            return team1_player, self.team1.color, str(team1_player.label)
        elif team2_player is not None:
            return team2_player, self.team2.color, str(team2_player.label)

    def resolve_team_helper(self, id: int, detection_info: Tuple[float, float], detection_info_meters: Tuple[float, float], team: int, jersey_number: int, name: str):
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
            self.team1.players.append(
                Player(name, jersey_number, id, detection_info, detection_info_meters))
        elif team == 1:
            self.team2.players.append(
                Player(name, jersey_number, id, detection_info, detection_info_meters))
        elif team == 2:
            self.referee = Referee(
                name, id, constants.YELLOW, detection_info, detection_info_meters)

        if team < 3:  # team1, team2 or referee
            self.initial_ids.add(id)
            return True
        return False

    def get_new_objects(self, bb_infos: Tuple[int, int, int, int], object_ids: List[int], detections_in_pitch: List[Tuple[int, int]]):
        """Returns new objects for the current frame based on the information on ignoring ids and teams.

        Args:
            bb_infos (Tuple[int, int, int, int]): Bounding boxes. 
            object_ids (List[int]): List of object identifiers.
            detections_in_pitch (Tuple[int, int]): All 2D detections that are in the pitch.
        Returns: Tuple of lists
        """
        new_object_detections: List[Tuple[int, int]] = []
        new_object_bb: List[Tuple[int, int, int, int]] = []
        new_object_ids: List[int] = []
        for i, obj_id in enumerate(object_ids):
            if obj_id not in self.ignore_ids and self.find_person_with_id(obj_id) is None:
                new_object_detections.append(detections_in_pitch[i])
                new_object_bb.append(bb_infos[i])
                new_object_ids.append(obj_id)
        return new_object_detections, new_object_bb, new_object_ids

    def resolve_user_action_helper(self, prompt: str, obj_id: int, det: Tuple[int, int], resolving_positions_cache: Dict[int, int], new_obj_ids: List[int], existing_obj_ids: List[int], resolve_helper: resolve_helpers.LastNFramesHelper, prompter):
        """Helper action used when needed recursion to resolve user action.

        Args:
            prompt (str): Prompt that will be given to the thread
            obj_id (int): Object id 
            det (Tuple[int, int]): Detection in 2D space.
            resolving_positions_cache (Dict[int, int]): For saving user actions.
            new_obj_ids (List[int]): All new objects in the current frame.
            existing_obj_ids (List[int]): All objects that exist from the previous frame.
            resolve_helper (resolve_helpers.LastNFramesHelper): Visualizatio helper engine.
            prompter (_type_): Thread prompter
        """
        prompter.set_execution_config(prompt)
        resolve_helper.visualize(constants.DETECTIONS_WINDOW, obj_id, det)
        new_action = int(prompter.value)
        self.resolve_user_action(new_action, obj_id, det, resolving_positions_cache,
                                 new_obj_ids, existing_obj_ids, resolve_helper, prompter)

    def resolve_user_action(self, action: int, obj_id: int, det: Tuple[int, int], resolving_positions_cache: Dict[int, int], new_obj_ids: List[int], existing_obj_ids: List[int], resolve_helper: resolve_helpers.LastNFramesHelper, prompter):
        """Resolves user input based on a few simple conditions. Don't allow adding new id by yourself, only YOLO can generate new id.

        Args:
            action (int): Action that user requested.
            obj_id (int): Id of the object that needs resolving
            det (Tuple[int, int]): Detection in 2D space in pixels.
            resolving_positions_cache (Dict[int, int]): For saving user actions.
            new_obj_ids (List[int]): All new objects in the current frame.
            existing_obj_ids (List[int]): All objects that exist from the previous frame.
            resolve_helper (resolve_helpers.LastNFramesHelper): Instance of the helper.
            prompter Reference to the prompter thread
        """
        resolving_positions_cache[obj_id] = action  # if recursively call, it will be overwritten
        if action == 0:
            self.referee.ids.append(obj_id)
        elif action == -1:
            self.ignore_ids.append(obj_id)  # from now on ignore this id
        elif action == -2:
            self.resolve_user_action_helper(constants.prompt_input, obj_id, det, resolving_positions_cache, new_obj_ids,
                                            existing_obj_ids, resolve_helper, prompter)
        else:
            # Don't allow assigning id to the player that is already shown in the frame
            if action not in new_obj_ids and action in existing_obj_ids:
                self.resolve_user_action_helper(f"Please insert again, person with id {action} is already shown in the current frame", obj_id, det,
                                                resolving_positions_cache, new_obj_ids, existing_obj_ids, resolve_helper, prompter)
            else:
                ex_player = self.find_player_with_id(action)
                if ex_player is not None:
                    ex_player.ids.append(obj_id)
                else:
                    self.resolve_user_action_helper(f"Please insert again, person with id {action} doesn't exist.", obj_id, det, resolving_positions_cache,
                                                    new_obj_ids, existing_obj_ids, resolve_helper, prompter)

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
                    players_data[int(line_data[0])] = (
                        line_data[1], (line_data[2], line_data[3]))
        except Exception:
            return False

        match = Match(team1_name, team2_name)
        for obj_id, (line, initial_position) in players_data.items():
            initial_position = utils.to_tuple_float(initial_position)
            values = list(
                map(lambda val: val.strip().rstrip(), line.split(',')))
            team = int(values[0])
            jersey_number = int(values[1])
            if not match.resolve_team_helper(obj_id, initial_position, pitch.pixel_to_meters_positions(initial_position), team, jersey_number, values[2]):
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
            A reference to the match.
        """
        print("Please insert the name of the first team: ")
        team_name1 = input()
        print("Please insert the name of the second team: ")
        team_name2 = input()
        match = Match(team_name1, team_name2)
        player_cache: Tuple[int, Tuple[str, Tuple[int, int]]] = dict()
        for i, bb_obj_info in enumerate(bb_info):
            show_frame = img.copy()
            view.View.box_label(show_frame, bb_obj_info,
                                constants.BLACK, obj_ids[i])
            while True:
                prompter.set_execution_config(
                    "Insert 0 for the first team, 1 for the second and 2 for the referee, jersey_number or -1, name: ")
                view.View.show_img_while_not_killed([window], [show_frame])
                value = prompter.value
                values = prompter.value.split(',')
                values = list(map(lambda val: val.strip().rstrip(), values))
                team = int(values[0])
                jersey_number = int(values[1])
                try:
                    if not match.resolve_team_helper(obj_ids[i], detections_info[i], pitch.pixel_to_meters_positions(detections_info[i]), team, jersey_number, values[2]):
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
                file.write(
                    f"{obj_id}###{line}###{initial_position[0]}###{initial_position[1]}\n")
        return match
