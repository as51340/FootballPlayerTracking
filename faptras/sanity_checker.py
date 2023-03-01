from typing import Dict, List, Tuple
from collections import Counter

class SanityChecker:
    
    def check(self, objects_ids_to_draw: List[int]):
        """Checks if there is any ids that occurs multiple times.

        Args:
            objects_ids_to_draw (List[int]): 
        """
        res = Counter(objects_ids_to_draw)
        for id_, num_occus in res.items():
            if num_occus > 1:
                print(f"Multiple instance of object with id: {id_} {objects_ids_to_draw}")
    
    
    def clear_already_resolved(self, detections_in_pitch: List[Tuple[int, int]], bb_info_in_pitch: List[Tuple[int, int, int, int]], objects_id_in_pitch: List[int], resolving_positions_cache: Dict[int, int]):
        """Removes information about objects that were already resolved based on the resolving positions cache.

        Args:
            detections_in_pitch (List[Tuple[int, int]]): Detections inside the pitch.
            bb_info_in_pitch (List[Tuple[int, int, int, int]]): Bounding box for objects inside the pitch.
            objects_id_in_pitch (List[int]): Ids of the objects inside the pitch.
            resolving_positions_cache (Dict[int, int]): Cache: old id to new id

        Returns:
            Detections, bboxes and ids of cleaned objects.
        """
        to_delete = []
        for i in range(len(objects_id_in_pitch)):
            if objects_id_in_pitch[i] in resolving_positions_cache.keys():
                to_delete.append(i)
        detections_in_pitch = [det for i, det in enumerate(detections_in_pitch) if i not in to_delete]
        bb_info_in_pitch = [bb for i, bb in enumerate(bb_info_in_pitch) if i not in to_delete]
        objects_id_in_pitch = [id_ for i, id_ in enumerate(objects_id_in_pitch) if i not in to_delete]
        return detections_in_pitch, bb_info_in_pitch, objects_id_in_pitch