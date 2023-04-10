from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import time
from collections import Counter

from abc import abstractmethod, ABC


class TeamClassificator(ABC):

    @abstractmethod
    def classify_persons_into_categories(self, detections: np.ndarray, max_height=-1, max_width=-1):
        pass

    def preprocess_detections(self, detections: np.ndarray):
        return detections.reshape(detections.shape[0], -1)


class DBSCANTeamClassificator(TeamClassificator):

    def __init__(self) -> None:
        self.alg = DBSCAN(eps=3, min_samples=2)

    def classify_persons_into_categories(self, detections: np.ndarray, max_height=-1, max_width=-1):
        start_time = time.time()
        labels = self.alg.fit_predict(self.preprocess_detections(detections))
        print(f"Labels: {labels}")
        print(f"Fit and predict took: {time.time() - start_time:.2f}s")
        return labels


class KMeansTeamClassificator(TeamClassificator):

    def __init__(self) -> None:
        self.alg = KMeans(n_clusters=3)

    def classify_persons_into_categories(self, detections: np.ndarray, max_height=-1, max_width=-1):
        """Tries to classify objects into two teams and a referee using clustering algorithms.

        Args:
            detections (np.ndarray): N*H*W*C 
            max_height (int): _description_
            max_width (int): _description_

        Returns:
            _type_: _description_
        """
        start_time = time.time()
        # Detections
        # if self.detections is None:
        #     self.detections = detections
        # else:
        #     self.detections = np.vstack((self.detections, detections))
        # print(f"Shape of detections: {detections.shape} {self.detections.shape}")
        # Transfer because you will need it for padding
        # if max_height > self.max_height:
        #     self.max_height = max_height
        # if max_width > self.max_width:
        #     self.max_width = max_width
        detections_ = self.preprocess_detections(detections)
        self.alg.fit(detections_)
        labels = self.alg.predict(detections_)
        print(f"Labels: {labels}")
        print(f"Fit and predict took: {time.time() - start_time:.2f}s")
        return labels


# def get_bounding_boxes(bb_info, video_frame):
#     bboxes = []
#     max_height, max_width = -1, -1
#     for bb in bb_info:
#         bbox = video_frame[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], :]  # H*W*C
#         if bb[3] > max_height:
#             max_height = bb[3]
#         if bb[2] > max_width:
#             max_width = bb[2]
#         bboxes.append(bbox)
#     print(f"Max height, width: {max_height}, {max_width}")
#     for i, bbox in enumerate(bboxes):
#         lpad, upad = int((max_width - bbox.shape[1]) / 2), int((max_height - bbox.shape[0]) / 2)
#         rpad, dpad = lpad + int((max_width - bbox.shape[1]) % 2), upad + int((max_height - bbox.shape[0]) % 2)
#         # print(f"bbox shape: {bbox.shape}")
#         # print(f"l, u, r, d: {lpad} {upad} {rpad} {dpad}")
#         bboxes[i] = np.pad(bbox, ((upad, dpad), (lpad, rpad), (0, 0)), mode="mean")
#         # print(f"After padding shape: {bboxes[i].shape}")
#
#     bboxes = np.array(bboxes)
#     #`` print(f"All boxes shape: {bboxes.shape}")
#     return bboxes, max_height, max_width


# Should have been used
# def is_assistant_referee_positioned(pitch: Pitch, detection_x_coord: int, detection_y_coord: int) -> bool:
#     """Checks whether the detection is positioned as a assistant referee. This can be side-line referee but also 4th behind the goal.
#
#     Args:
#         pitch (Pitch): _description_
#         detection_x_coord (int): _description_
#         detection_y_coord (int): _description_
#
#     Returns:
#         bool: true if it is positioned like the assistant referee could be positioned.
#     """
#     # Must not be with both coordinates outside of the pitch.
#     if detection_x_coord >= pitch.upper_left_corner[0] - config.ASSISTANT_REFEREE_PIXEL_TOLERANCE and detection_x_coord <= pitch.upper_left_corner[0] + config.ASSISTANT_REFEREE_PIXEL_TOLERANCE and \
#         detection_y_coord >= pitch.upper_left_corner[1] and detection_y_coord <= pitch.down_left_corner[1]:
#             return True  # left sideline check
#     if detection_x_coord >= pitch.upper_right_corner[0] - config.ASSISTANT_REFEREE_PIXEL_TOLERANCE and detection_x_coord <= pitch.upper_right_corner[0] + config.ASSISTANT_REFEREE_PIXEL_TOLERANCE and \
#         detection_y_coord >= pitch.upper_right_corner[1] and detection_y_coord <= pitch.down_right_corner[1]:
#             return True  # right sideline check
#     if detection_x_coord >= pitch.upper_left_corner[0] and detection_x_coord <= pitch.upper_right_corner[0] and \
#         detection_y_coord >= pitch.upper_left_corner[1] - config.ASSISTANT_REFEREE_PIXEL_TOLERANCE and detection_y_coord <= pitch.upper_left_corner[1] + config.ASSISTANT_REFEREE_PIXEL_TOLERANCE:
#             return True  # up sideline check
#     if detection_x_coord >= pitch.down_left_corner[0] and detection_x_coord <= pitch.down_right_corner[0] and \
#         detection_y_coord >= pitch.down_left_corner[1] - config.ASSISTANT_REFEREE_PIXEL_TOLERANCE and detection_y_coord <= pitch.down_left_corner[1] + config.ASSISTANT_REFEREE_PIXEL_TOLERANCE:
#             return True  # down sideline check
#     return False
