from typing import List, Dict, Tuple

import match


class Resolver:
    
    def resolve(self, match_: match.Match, new_object_ids: List[int], existing_ids_in_frame: List[int]):
        # If we have only one new object and all other from the beginning are being successfully tracked except this 1
        # This is ok under the assumption that we are tracking referee and all players from both teams
        if len(new_object_ids) == 1:
            # Find the missing player
            missing_ids_from_start = [id_ for id_ in match_.initial_ids if id_ not in existing_ids_in_frame]
            if len(missing_ids_from_start) == 1:
                # It means we found that player
                print(f"Automatic resolvement of id {new_object_ids[0]} to {missing_ids_from_start[0]}")
                return missing_ids_from_start[0]
            return missing_ids_from_start