from typing import List, Tuple, Dict
import os
from collections import defaultdict

import numpy as np
import cv2 as cv

# Constants
RED = (0, 0, 255)
SRC_WINDOW, DST_WINDOW, DETECTIONS_WINDOW = "src", "dst", "det"

# Global variables
last_clicked_windows = []


# mouse callback function
def _select_points_wrapper(event, x, y, _, params):
    """Wrapper for mouse callback.
    """
    global last_clicked_window
    window, points, img_copy = params
    if event == cv.EVENT_LBUTTONDOWN:
        last_clicked_windows.append(window)
        cv.circle(img_copy, (x,y), 5, RED, -1)
        points.append([x, y])

def read_image(window: str, img_path: str, points_storage: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Reads image from the given path and sets mouse callback.
    Args:
        img_path (str): Path of the image.
        window (str): Windows' names to create -> two are created, one for real image and the other for the artificial image.
        points_storage (str): Storage where points will be saved on a mouse click.
    """
    # Registers a mouse callback function on a image's copy.
    img = cv.imread(img_path, -1)
    img_copy = img.copy()
    cv.namedWindow(window)
    cv.moveWindow(window, 80,80);
    cv.setMouseCallback(window, _select_points_wrapper, (window, points_storage, img_copy))
    return img, img_copy

def get_plan_view(src_img: np.ndarray, dst_img: np.ndarray, src_points: List[List[int]], dst_points: List[List[int]]):
    """Warps original image.
    Args:
        src_points (List[List[int]]): Source points' storage.
        dst_points (List[List[int]]): Destination points' storage.
        src_img (numpy.ndarray): Source image representation.
        dst_img (numpy.ndarray): Destination image representation.
    Returns:
        warped image, homography matrix and the mask.
    """
    src_pts = np.array(src_points).reshape(-1,1,2)
    dst_pts = np.array(dst_points).reshape(-1,1,2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    H = np.array(H)
    plan_view = cv.warpPerspective(src_img, H, (dst_img.shape[1], dst_img.shape[0]))
    return plan_view, H, mask

def merge_views(src_img: np.ndarray, dst_img: np.ndarray, src_points: List[List[int]], dst_points: List[List[int]]):
    """Merges two views.
    Args:
        src_points (List[List[int]]): Source points' storage.
        dst_points (List[List[int]]): Destination points' storage.
        src_img (np.ndarray): Source image.
        dst_img (np.ndarray): Destination image.
    """
    plan_view, H, mask = get_plan_view(src_img, dst_img, src_points, dst_points)
    for i in range(0, dst_img.shape[0]):
        for j in range(0, dst_img.shape[1]):
            if plan_view[i, j, 0] == 0 and plan_view[i, j, 1] == 0 and plan_view[i, j, 2] == 0:
                plan_view[i, j, 0] = dst_img[i, j, 0]
                plan_view[i, j, 1] = dst_img[i, j, 1]
                plan_view[i, j, 2] = dst_img[i, j, 2]
    return plan_view, H, mask

def visualize(src_points: List[List[int]], dst_points: List[List[int]], src_img: np.ndarray, src_img_copy: np.ndarray, dst_img: np.ndarray, dst_img_copy: np.ndarray):
    """Runs visualization in while loop.
    Args:
        src_points (List[List[int]]): Source points' storage.
        dst_points (List[List[int]]): Destination points' storage.
        src_img (numpy.ndarray): Source image representation.
        src_img_copy (numpy.ndarray): Copy of the source image representation.
        dst_img (numpy.ndarray): Destination image representation.
        dst_img_copy (numpy.ndarray): Copy of the destination image representation.
    """
    while(1):
        # Keep showing copy of the image because of the circles drawn
        cv.imshow(SRC_WINDOW, src_img_copy)
        cv.imshow(DST_WINDOW, dst_img_copy)
        k = cv.waitKey(1) & 0xFF
        if k == ord('h'):
            print('create plan view')
            plan_view, H, mask = get_plan_view(src_img, dst_img, src_points, dst_points)
            cv.imshow("plan view", plan_view) 
        elif k == ord('m'):
            print('merge views')
            merge = merge_views(src_img, dst_img, src_points, dst_points)      
            cv.imshow("merge", merge)        
        elif k == ord('d') and last_clicked_windows:
            undo_window = last_clicked_windows.pop()
            if undo_window == SRC_WINDOW and src_points:
                src_points.pop()
                # Reinitialize image
                src_img_copy = src_img.copy()
                cv.setMouseCallback(SRC_WINDOW, _select_points_wrapper, (SRC_WINDOW, src_points, src_img_copy))
                draw_old_circles(src_img_copy, src_points)
            elif undo_window == DST_WINDOW and dst_points:
                dst_points.pop()
                # Reinitialize image
                dst_img_copy = dst_img.copy()
                cv.setMouseCallback(DST_WINDOW, _select_points_wrapper, (DST_WINDOW, dst_points, dst_img_copy))
                draw_old_circles(dst_img_copy, dst_points)
        elif k == ord("s"):
            print(f"Source points: {src_points}")
            print(f"Dest points: {dst_points}")
        elif k == ord('q'):
            cv.destroyAllWindows()
            return plan_view, H, mask

def draw_old_circles(img, points):
    """Draws all saved points on an image. Used for implementing undo buffer.
    Args:
        img (np.ndarray): A reference to the image.
        points (List[List[int]]): Points storage.
    """
    for x, y in points:
        cv.circle(img, (x,y), 5, RED, -1)


def squash_detections(path_to_detections: str, H: np.ndarray, squasher_func, info_extract_func):
    """Squashes detections from the bounding box to the one point and transforms them using the homography matrix.
    Args:
        path_to_detections (str): 
        H (np.ndarray): Homography matrix.
        squasher_func: Lambda function to transform bounding boxes into one point.

    """
    with open(path_to_detections, "r") as d_file:
        squashed_temp_detections = []
        squashed_detections = []
        cnt = 0
        lines = d_file.readlines()
        squashed_detections = list(map(squasher_func, lines)) # ...
        for line in lines:
            if cnt == 10:
                break
            line = line.split(" ")
            # Extract bounding box information
            frame_id = int(line[0])
            object_id = int(line[1])
            bb_left = float(line[2])
            bb_top = float(line[3])
            bb_width = float(line[4])
            bb_height = float(line[5])
            # Calculate lower center
            bb_lower_center = [bb_left + 0.5 * bb_width, bb_top + bb_height, 1]  # for (in)homogeneous coordinates
            print(f"BB lower center: {bb_lower_center}")
            squashed_temp_detections.append((frame_id, object_id))
            squashed_detections.append(bb_lower_center)
            cnt += 1
        squashed_detections = np.array(squashed_detections)
        print(f"Before transformation: {squashed_detections}")
        squashed_detections = squashed_detections@H
        print(f"After transformation: {squashed_detections}")
        return squashed_detections

def draw_detections(detections: Dict[int, List[float]], pitch_img):
    for frame_id, squashed_detections in detections.items():
        print(f"Frame id: {frame_id}")
        cv.imshow(DETECTIONS_WINDOW, pitch_img)
        # for squashed_detection in squashed_detection:
         #    cv.circle(pitch_img, (x, y), 5, RED, -1)


if __name__ == "__main__":
    # Path to images
    SRC_IMG_PATH = os.path.join(os.getcwd(), "./images/materials/reference_img.jpg")
    DST_IMG_PATH = os.path.join(os.getcwd(), "./images/materials/pitch.jpg")
    print(f"Real image: {SRC_IMG_PATH}")
    print(f"Artificial image: {DST_IMG_PATH}")
    # Path to detections
    DET_PATH = os.path.join(os.getcwd(), "./results/track/exp17/tracks/t4.txt")
    print(f"Path to detections: {DET_PATH}")
    # Setup homography
    src_points, dst_points = [], []
    src_img, src_img_copy = read_image(SRC_WINDOW, SRC_IMG_PATH, src_points)
    dst_img, dst_img_copy = read_image(DST_WINDOW, DST_IMG_PATH, dst_points)

    # plan_view, H, mask = visualize(src_points, dst_points, src_img, src_img_copy, dst_img, dst_img_copy)

    H = np.array([[4.96880829e-03, 1.38109006e+00, -9.50897851e+01],
                [-3.33220488e-01, 4.19221488e-01, 9.04820655e+02],
                [1.35525043e-04, 2.46362905e-03, 1.00000000e+00]]

    print(f"H type: {type(H)}")

    bv_lower_centers = squash_detections(DET_PATH, H)