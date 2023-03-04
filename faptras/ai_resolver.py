from typing import List, Dict, Tuple

import match
from person import Person
import utils
import constants


class ResolvingInfo:
    
    def __init__(self) -> None:
        self.found_ids: List[int] = []
        self.resolved_ids: List[int] = []  # new object id that was resolved
        self.unresolved_ids: List[int] = []  # new object id
        self.resolved_detections: List[Tuple[int, int]] = []  # coordinates of objects that were resolved
        self.unresolved_detections: List[Tuple[int, int]] = []  # coordinates of objects that were not resolved
        self.resolved_bbs: List[Tuple[int, int, int, int]] = []  # resolved bounding boxes
        self.unresolved_bbs: List[Tuple[int, int, int, int]] = []  # unesolved bounding boxes
        self.unresolved_starting_ids: List[int] = []  # all is that are missing from the start but were not found

        
class Resolver:
    
    def __init__(self, fps: int) -> None:
        """

        Args:
            fps (int): FPS of the original video.
        """
        self.fps = fps
    
    def resolve(self, pitch_, match_: match.Match, new_objects_detection, new_objects_bb, new_objects_id: List[int], existing_objects_detection, existing_objects_bb, existing_objects_id: List[int], frame_id: int):
        """

        Args:
            pitch_ (_type_): _description_
            match_ (match.Match): _description_
            new_objects_detection (_type_): _description_
            new_objects_id (List[int]): _description_
            existing_objects_detection (_type_): _description_
            existing_objects_id (List[int]): _description_
            frame_id (int): _description_

        Returns:
            _type_: Unresolved detections and unresolved bounding boxes.
        """
        # If we have only one new object and all other from the beginning are being successfully tracked except this 1
        # This is ok under the assumption that we are tracking referee and all players from both teams
        print(f"Started resolvement process in frame {frame_id}")
        result = ResolvingInfo()
        
        # All ids that are in the match from the start but not in the known objects
        missing_ids_from_start = [id_ for id_ in match_.initial_ids if id_ not in existing_objects_id]
        if len(new_objects_id) == 1 and len(missing_ids_from_start) == 1:
            # It means we found that player
            print(f"Auto-resolve of object {new_objects_id[0]} to {missing_ids_from_start[0]}.")
            result.found_ids.append(missing_ids_from_start[0]) 
            result.resolved_ids.append(new_objects_id[0])
            result.resolved_detections.append(new_objects_detection[0])
            result.resolved_bbs.append(new_objects_bb[0])
            return result

        # If not go to more complex resolvement step
        # First, for all missing persons from the start of the match, calculate for how long they haven't been seen
        # One unneccessary fin
        missing_persons: List[Person] = []
        missing_persons_time_passed: List[float] = []
        for i in range(len(missing_ids_from_start)):
            person = match_.find_person_with_id(missing_ids_from_start[i])
            time_sec = (frame_id - person.last_seen_frame_id) / self.fps
            missing_persons_time_passed.append(time_sec)
            missing_persons.append(person)
        
        unresolved_starting_ids: List[int] = missing_ids_from_start.copy()
        
        # Not iterate over all new objects
        for j in range(len(new_objects_id)):
            # Filter out those that are too fast
            distances: List[float] = []
            possible_indices: List[int] = []  # Possible indices for this new_object_id
            min_dist, min_ind, min_avg_dist, min_avg_ind = None, None, None, None
            for i in range(len(missing_ids_from_start)):
                if missing_persons_time_passed[i] < 0.2:  # ignore less than 0.2s
                    print(f"Ignoring calculation for player {missing_ids_from_start[i]} and new player {new_objects_id[j]} because of too short time passed: {missing_persons_time_passed[i]:.2f}s.")
                elif missing_persons_time_passed[i] > 5:
                    print(f"Ignoring calculation for player {missing_ids_from_start[i]} and new player {new_objects_id[j]} because of too long time passed: {missing_persons_time_passed[i]:.2f}s.")
                else:
                    dist_meters = utils.calculate_euclidean_distance(missing_persons[i].current_position, pitch_.pixel_to_meters_positions(new_objects_detection[j]))
                    speed = dist_meters / missing_persons_time_passed[i]
                    if speed > constants.MAX_SPEED:
                        print(f"Player {missing_ids_from_start[i]} is certainly not {new_objects_id[j]} because he would have to run with {speed:.2f} m/s "
                              + f"considering his distance of {dist_meters:.2f}m from the object {new_objects_id[j]}.")
                        continue
                    print(f"Player {missing_ids_from_start[i]} could be {new_objects_id[j]} because he would have to run with {speed:.2f} m/s considering his distance "
                        + f"of {dist_meters:.2f}m from the object {new_objects_id[j]}.")
                    distances.append(dist_meters)
                    possible_indices.append(i)
                    # Criteria based on the minimal distance from the new detection
                    if min_dist is None or dist_meters < min_dist:
                        min_dist, min_ind = dist_meters, i
                    # Criteria based on the minimal average distance from the missing player's average position
                    person_avg_pos_x = missing_persons[i].sum_pos_x / missing_persons[i].total_times_seen  # avg pos x per frame
                    person_avg_pos_y = missing_persons[i].sum_pos_y / missing_persons[i].total_times_seen  # avg pos y per frame
                    avg_dist = utils.calculate_euclidean_distance((person_avg_pos_x, person_avg_pos_y), pitch_.pixel_to_meters_positions(new_objects_detection[j]))
                    if min_avg_dist is None or avg_dist < min_avg_dist:
                        min_avg_dist, min_avg_ind = avg_dist, i
            # Check if we maybe solved our problem
            if len(possible_indices) == 1:
                print(f"Auto-resolve of id {new_objects_id[j]} to {missing_ids_from_start[possible_indices[0]]} based on inference logic.")
                result.found_ids.append(missing_ids_from_start[possible_indices[0]])
                result.resolved_ids.append(new_objects_id[j])
                result.resolved_detections.append(new_objects_detection[j])
                result.resolved_bbs.append(new_objects_bb[j])
                unresolved_starting_ids.remove(missing_ids_from_start[possible_indices[0]])
            else: # We go to into the voting process
                # Check based on the minimal distance
                if min_dist is not None:
                    print(f"Based on min. distance, vote goes to: {missing_ids_from_start[min_ind]} for being {min_dist:.2f}m far from the object {new_objects_id[j]}.")
                    print(f"Distances from other missing players to object {new_objects_id[j]} are {distances}")
                else:
                    print(f"Cannot use minimal distance criteria.")
                # Check based on the average distance
                if min_avg_dist is not None:
                    print(f"Based on min. avg. distance, vote goes to: {missing_ids_from_start[min_avg_ind]} for being {min_avg_dist:.2f}m far from the object {new_objects_id[j]}")
                else:
                    print(f"Cannot use minimal average distance criteria.")
                # Check if they match together
                if min_ind == min_avg_ind and min_ind is not None:
                    print(f"Both conditions are met so we conclude that {new_objects_id[j]} is the person {missing_ids_from_start[min_ind]}")
                    # You have timeout to do it differently
                    result.found_ids.append(missing_ids_from_start[min_ind])
                    result.resolved_ids.append(new_objects_id[j])
                    result.resolved_detections.append(new_objects_detection[j])
                    result.resolved_bbs.append(new_objects_bb[j])
                    unresolved_starting_ids.remove(missing_ids_from_start[min_ind])
                else:
                    print(f"Resolver couldn't conclude automatically which player is shown so you will have to do manual resolvement")
                    result.unresolved_ids.append(new_objects_id[j])
                    result.unresolved_detections.append(new_objects_detection[j])
                    result.unresolved_bbs.append(new_objects_bb[j])
        print(f"Ended resolvement process in frame {frame_id}")
        result.unresolved_starting_ids = unresolved_starting_ids
        return result        