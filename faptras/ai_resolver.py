from typing import List, Dict, Tuple

import match
from person import Person
import utils
import constants
class Resolver:
    
    def __init__(self, fps: int) -> None:
        """

        Args:
            fps (int): FPS of the original video.
        """
        self.fps = fps
    
    def resolve(self, pitch_, match_: match.Match, new_object_ids: List[int], new_objects_position: List[int], existing_ids_in_frame: List[int], frame_id: int):
        # If we have only one new object and all other from the beginning are being successfully tracked except this 1
        # This is ok under the assumption that we are tracking referee and all players from both teams
        missing_ids_from_start = [id_ for id_ in match_.initial_ids if id_ not in existing_ids_in_frame]
        if len(new_object_ids) == 1 and len(missing_ids_from_start) == 1:
            # It means we found that player
            print(f"Automatic resolvement of id {new_object_ids[0]} to {missing_ids_from_start[0]}")
            return missing_ids_from_start[0]

        missing_persons: List[Person] = []
        possible_indices_based_on_time: List[int] = []  # all indices t
        missing_persons_time_passed: List[float] = []
        for i in range(len(missing_ids_from_start)):
            person = match_.find_person_with_id(missing_ids_from_start[i])
            missing_persons.append(person)
            time_sec = (frame_id - person.last_seen_frame_id) / self.fps
            if time_sec < 0.2 or time_sec > 5:
                possible_indices_based_on_time.append(False)
            else:
                possible_indices_based_on_time.append(True)
            missing_persons_time_passed.append(time_sec)
            
        for j in range(len(new_object_ids)):
            # Filter out those that are too fast
            possible_indices: List[int] = []
            distances: List[float] = []
            for i in range(len(missing_ids_from_start)):
                dist_meters = utils.calculate_euclidean_distance(missing_persons[i].current_position, pitch_.pixel_to_meters_positions(new_objects_position[i]))
                speed = dist_meters / missing_persons_time_passed[i]
                if speed > constants.MAX_SPEED:
                    print(f"Player {missing_ids_from_start[i]} is certainly not {new_object_ids[j]} because he would have to run with {speed:.2f} m/s "
                          + f"considering his distance of {dist_meters:.2f}m")
                    continue
                if missing_persons_time_passed[i] < 0.2:  # ignore less than 0.2s
                    print(f"Ignoring calculation for player {missing_ids_from_start[i]} and new player {new_object_ids[j]} because of too short time passed: {frame_id - missing_persons[i].last_seen_frame_id} frames.")
                elif missing_persons_time_passed[i] > 5:
                    print(f"Ignoring calculation for player {missing_ids_from_start[i]} and new player {new_object_ids[j]} because of too long time passed:  {frame_id - missing_persons[i].last_seen_frame_id} frames.")
                else:
                    print(f"Player {missing_ids_from_start[i]} could be {new_object_ids[j]} because he would have to run with {speed:.2f} m/s considering his distance "
                           + f"of {dist_meters:.2f}m")
                possible_indices.append(i)
                distances.append(dist_meters)
            # Check if we maybe solved our problem
            if len(possible_indices) == 1:
                print(f"Automatic resolvement of id {new_object_ids[j]} to {missing_ids_from_start[0]}")
                return missing_ids_from_start[0]
            # If not, for all for which calculations are possible based on time we can try based on the shortest distance
            min_dist, min_index = None, None
            for i in range(len(possible_indices)):
                if min_dist is None or distances[possible_indices[i]] < min_dist:
                    min_dist, min_index = distances[possible_indices[i]], possible_indices[i]
            if min_dist is not None:
                print(f"Based on min. distance, vote goes to: {missing_ids_from_start[min_index]} for having dist: {min_dist:.2f}m. All distances are: {'m, '.join(distances)}")
            else:
                print(f"Cannot use minimal distance criteria.")
            # We can always do average position
            min_avg_dist, min_avg_index = None, None
            avg_distances = []
            for i in range(len(missing_ids_from_start)):
                person_avg_pos_x = missing_persons[i].sum_pos_x / missing_persons[i].total_frames  # avg pos x per frame
                person_avg_pos_y = missing_persons[i].sum_pos_y / missing_persons[i].total_frames  # avg pos y per frame
                avg_dist = utils.calculate_euclidean_distance((person_avg_pos_x, person_avg_pos_y), new_objects_position[i])
                if min_avg_dist is None or avg_dist < min_avg_dist:
                    min_avg_dist, min_avg_index = avg_dist, i
                avg_distances.append(str(round(avg_dist, 2)))
            if min_avg_dist is not None:
                print(f"Based on min. avg. distance, vote goes to: {missing_ids_from_start[min_avg_index]} for having dist: {min_avg_dist:.2f}m. All distances are: {'m,'.join(avg_distances)} ")
            else:
                print(f"Cannot use minimal average distance criteria.")
            # We can always do based on coloring
            # If all three conditions can be used then use it = use as much conditions as possible
        return missing_ids_from_start