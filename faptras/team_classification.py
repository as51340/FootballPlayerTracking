from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import time
from collections import Counter

class TeamClassificator:
    
    def __init__(self, type: str) -> None:
        self.detections = None # cache all detections. We expect of the clustering algorithm to become eventually better and better.
        self.type = type.lower()
        if self.type == "kmeans":
            self.alg = KMeans(n_clusters=3)
        elif self.type == "dbscan":
            self.alg = DBSCAN(eps=3, min_samples=2)
        
    def classify_persons_into_categories(self, detections: np.ndarray):
        start_time = time.time()
        if self.detections is None:
            self.detections = detections
        else:
            self.detections = np.vstack((self.detections, detections))
        print(f"Shape of detections: {detections.shape} {self.detections.shape}")
        # self.alg.fit(self.detections)
        self.alg.fit(self.detections)
        labels = self.alg.labels_[-detections.shape[0]:]
        print(f"Labels: {labels}")
        # print(f"Labels: {dict(Counter(labels))}")
        print(f"Refitting and predicting took: {time.time() - start_time}")
        return labels
        
    