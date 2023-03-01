from typing import Dict, List, Tuple
from collections import Counter

class SanityChecker:
    
    def check(self, objects_ids_to_draw: List[int]):
        res = Counter(objects_ids_to_draw)
        for id_, num_occus in res.items():
            if num_occus > 1:
                print(f"Multiple instance of object with id: {id_} {objects_ids_to_draw}")
                
    