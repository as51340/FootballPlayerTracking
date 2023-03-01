from typing import Dict, List, Tuple
from collections import Counter

class SanityChecker:
    
    def check(self, objects_ids_to_draw: List[int]):
        res = Counter(objects_ids_to_draw)
        for id_, num_occus in res.items():
            if num_occus > 1:
                print(f"Multiple instance of object with id: {id_} {objects_ids_to_draw}")
    
    
    def clear_already_resolved(self, detections_in_pitch, bb_info_in_pitch, object_ids_in_pitch, resolving_positions_cache: Dict[int, int]):
        to_delete = []
        for i in range(len(object_ids_in_pitch)):
            if object_ids_in_pitch[i] in resolving_positions_cache.keys():
                to_delete.append(i)
        detections_in_pitch = [det for i, det in enumerate(detections_in_pitch) if i not in to_delete]
        bb_info_in_pitch = [bb for i, bb in enumerate(bb_info_in_pitch) if i not in to_delete]
        object_ids_in_pitch = [id_ for i, id_ in enumerate(object_ids_in_pitch) if i not in to_delete]
        return detections_in_pitch, bb_info_in_pitch, object_ids_in_pitch