from typing import List, Tuple
import numpy as np
import cv2 as cv

import constants
import view


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

def do_homography(view_: view.View, detections_vid_capture, path_to_ref_img: str, path_to_pitch: str, homo_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper function that does all necessary preparation for calling calculate_homography function.

    Args:
        view_ (view.View): reference to the view_ object
        detections_vid_capture (_type_): reference to the video
        path_to_ref_img (str): 
        path_to_pitch (str): 
        homo_file (str): path where the homography file is saved.

    Returns:
        Tuple[np.ndarray, np.ndarray]: reference frame = 0th frame from th video and homography matrix.
    """
    _, reference_frame = take_reference_img_for_homography(detections_vid_capture, path_to_ref_img)
    # Create homography
    H = calculate_homography(view_, reference_frame, path_to_pitch)
    np.save(homo_file, H)
    return reference_frame, H

def calculate_homography(view_: view.View, src_img: np.ndarray, path_to_pitch: str):
    """Runs visualization in while loop.
    Args:
        src_points (List[List[int]]): Source points' storage.
        dst_points (List[List[int]]): Destination points' storage.
        src_img (numpy.ndarray): Source image representation.
        src_img_copy (numpy.ndarray): Copy of the source image representation.
        dst_img (numpy.ndarray): Destination image representation.
        dst_img_copy (numpy.ndarray): Copy of the destination image representation.
    """
    # Source window setup
    src_points, dst_points = [], []
    src_img_copy: np.ndarray = src_img.copy()
    dst_img = cv.imread(path_to_pitch, -1)
    dst_img_copy: np.ndarray = dst_img.copy()
    view.View.full_screen_on_monitor(constants.SRC_WINDOW)
    cv.setMouseCallback(constants.SRC_WINDOW, view_._select_points_wrapper, (constants.SRC_WINDOW, src_points, src_img_copy))
    # Destination window setup
    cv.namedWindow(constants.DST_WINDOW)
    cv.setMouseCallback(constants.DST_WINDOW, view_._select_points_wrapper, (constants.DST_WINDOW, dst_points, dst_img_copy))
    referee_id = -1
    while(1):
        # Keep showing copy of the image because of the circles drawn
        cv.imshow(constants.SRC_WINDOW, src_img_copy)
        cv.imshow(constants.DST_WINDOW, dst_img_copy)
        k = cv.waitKey(1) & 0xFF
        if k == ord('h'):
            print('create plan view')
            plan_view, H, _ = get_plan_view(src_img, dst_img, src_points, dst_points)
            cv.imshow("plan view", plan_view)
        elif k == ord('d') and view_.last_clicked_windows:
            undo_window = view_.last_clicked_windows.pop()
            if undo_window == constants.SRC_WINDOW and src_points:
                src_points.pop()
                # Reinitialize image
                src_img_copy = src_img.copy()
                cv.setMouseCallback(constants.SRC_WINDOW, view_._select_points_wrapper, (constants.SRC_WINDOW, src_points, src_img_copy))
                view.View.draw_old_circles(src_img_copy, src_points)
            elif undo_window == constants.DST_WINDOW and dst_points:
                dst_points.pop()
                # Reinitialize image
                dst_img_copy = dst_img.copy()
                cv.setMouseCallback(constants.DST_WINDOW, view_._select_points_wrapper, (constants.DST_WINDOW, dst_points, dst_img_copy))
                view.View.draw_old_circles(dst_img_copy, dst_points)
        elif k == ord("s"):
            print(f"Source points: {src_points}")
            print(f"Dest points: {dst_points}")
        elif k == ord('q'):
            cv.destroyAllWindows()
            return H

def take_reference_img_for_homography(vid_capture: cv.VideoCapture, reference_img_path: str):
    """Takes 0th frame from the video for using it as a reference image for creating homography.

    Args:
        vid_capture (cv.VideoCapture): Reference to the video
        reference_img_path (str): Path where to save the reference image.
    """
    ret, frame = vid_capture.read()
    if ret:
        cv.imwrite(reference_img_path, frame)
    else:
        print("Could not take reference img")
    return ret, frame

