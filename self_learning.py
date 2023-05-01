# -------------------------------------------------------------------------------------------------------------------------------------- #
# Script that is used to merge the data obtain from the tracker                                                                          #
# to append it to the new dataset. Used for implementing so called                                                                       #
# self-learning. How to run: python self_learning.py --path-to-new-dataset /home/andi/FER/year5/FootballPlayerTracking/data/original_t16 # 
# --path-to-tracking-data /home/andi/FER/year5/FootballPlayerTracking/results/track/exp105/tracks/t16.txt                                #
# --path-to-original-video /home/andi/FER/year5/FootballPlayerTracking/data/t16.mp4                                                      # 
# -------------------------------------------------------------------------------------------------------------------------------------- #


import argparse
import os
from pathlib import Path

import cv2 as cv


TRAIN_DATASET_RATIO = 0.9
VALID_DATASET_RATIO = 0.1
TEST_DATASET_RATIO = 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="FAPTRAS: Football analysis player tracking software. Self-learning script.",
                                     description="Software used for tracking players and generating statistics based on it.")
    parser.add_argument("--path-to-new-dataset", type=str, required=True,
                        help="Path to the new dataset that will be used for self-learning.")
    parser.add_argument("--path-to-tracking-data", type=str, required=True,
                        help="Path to the file in which tracking data is located.")
    parser.add_argument("--path-to-original-video", type=str, required=True,
                        help="Path to the video for which tracking data located in --path-to-tracking-data is obtained")
    args = parser.parse_args()
    # Original video
    video_capture = cv.VideoCapture(args.path_to_original_video)
    width_orig = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height_orig = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_name = args.path_to_original_video.split("/")[-1].split(".")[0]
    dataset_name = args.path_to_new_dataset.split("/")[-1].split(".")[0]
    print(f"Video name, video size: {video_name}, {width_orig} {height_orig}")
    # Tracking data
    with open(args.path_to_tracking_data, "r") as tracking_file:
        lines = tracking_file.readlines()
        # Determine the number of entries that will be added to train, validation and test dataset
        num_tracking_data = len(lines)
        tracking_train_frames = int(TRAIN_DATASET_RATIO * num_tracking_data)
        tracking_valid_frames = int(VALID_DATASET_RATIO * num_tracking_data)
        tracking_test_frames = num_tracking_data - \
            tracking_train_frames - tracking_valid_frames
        # Determine the path to the original train, validation and test dataset
        train_images_path = os.path.join(
            args.path_to_new_dataset, "train", "images")
        train_labels_path = os.path.join(
            args.path_to_new_dataset, "train", "labels")
        valid_images_path = os.path.join(
            args.path_to_new_dataset, "valid", "images")
        valid_labels_path = os.path.join(
            args.path_to_new_dataset, "valid", "labels")
        test_images_path = os.path.join(
            args.path_to_new_dataset, "test", "images")
        test_labels_path = os.path.join(
            args.path_to_new_dataset, "test", "labels")
        Path(args.path_to_new_dataset).mkdir(parents=True, exist_ok=True)
        Path(train_images_path).mkdir(parents=True, exist_ok=True)
        Path(train_labels_path).mkdir(parents=True, exist_ok=True)
        Path(valid_images_path).mkdir(parents=True, exist_ok=True)
        Path(valid_labels_path).mkdir(parents=True, exist_ok=True)
        Path(test_images_path).mkdir(parents=True, exist_ok=True)
        Path(test_labels_path).mkdir(parents=True, exist_ok=True)
        # Iterate over each lne in the tracking data file
        last_frame = None
        first_line = lines[0].split(" ")
        first_frame = int(first_line[0])
        video_capture.set(cv.CAP_PROP_POS_FRAMES, int(first_frame) - 1)
        
        # Create data.yaml file
        data_file_path = os.path.join(args.path_to_new_dataset, "data.yaml")
        with open(data_file_path, "w") as data_file:
            data_file.write(f"train: ./{dataset_name}/train/images\n")
            data_file.write(f"val: ./{dataset_name}/valid/images\n")
            data_file.write(f"test: ./{dataset_name}/test/images\n\n")
            data_file.write("nc: 2\n")
            data_file.write("names: ['person', 'ball']")

        for index in range(num_tracking_data):
            line = lines[index].split(" ")
            # Determine frame to which we need to position
            frame = line[0]
            if int(frame) % 50 == 0:
                print(
                    f"Processed {int(frame) / video_capture.get(cv.CAP_PROP_FPS)} seconds")
            class_id = line[1]
            # Get coordinates
            bb_left = float(line[3])
            bb_top = float(line[4])
            bb_width = float(line[5])
            bb_height = float(line[6])
            bb_center_x = bb_left + 0.5 * bb_width
            bb_center_y = bb_top + 0.5 * bb_height
            bb_center_x_norm = bb_center_x / width_orig
            bb_center_y_norm = bb_center_y / height_orig
            bb_width_norm = bb_width / width_orig
            bb_height_norm = bb_height / height_orig
            # Add to the set so we don't save same frame image multiple timesw
            if index < tracking_train_frames:
                img_path = os.path.join(
                    train_images_path, frame.zfill(8)+".jpg")
                labels_path = os.path.join(
                    train_labels_path, frame.zfill(8)+".txt")
            elif index >= tracking_train_frames and index < tracking_train_frames + tracking_valid_frames:
                img_path = os.path.join(
                    valid_images_path, frame.zfill(8)+".jpg")
                labels_path = os.path.join(
                    valid_labels_path, frame.zfill(8)+".txt")
            else:
                img_path = os.path.join(
                    test_images_path, frame.zfill(8)+".jpg")
                labels_path = os.path.join(
                    test_labels_path, frame.zfill(8)+".txt")
            if frame != last_frame:
                cv.imwrite(img_path, video_capture.read()[1])
                last_frame = frame
            with open(labels_path, "a+") as labels_file:
                labels_file.write(
                    f"{class_id} {bb_center_x_norm} {bb_center_y_norm} {bb_width_norm} {bb_height_norm}\n")
        
