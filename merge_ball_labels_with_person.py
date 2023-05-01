# ----------------------------------------------------------------------------- #
# Appends ball labels to the old person labels.                                 #
# TODO: Introduce argparse to format arguments to the script.                   #
# ----------------------------------------------------------------------------- #


import os
import shutil

def merge(ball_labels_folder: str, old_labels_folder: str):
    """Important: Don't call it more than once. It will append the ball labels to the old labels.

    Args:
        ball_labels_folder (str): _description_
        old_labels_folder (str): _description_
    """
    for ball_label_file_name in os.listdir(ball_labels_folder):
        # Full ball label file path
        ball_label_file_path = os.path.join(ball_labels_folder, ball_label_file_name)
        print(f"Full path: {ball_label_file_path} File name: {ball_label_file_name}")
        
        with open(ball_label_file_path, "r") as ball_label_file:
            # Read ball annotation file
            label_line = ball_label_file.readline()
            label_line = '1' + label_line[1:]
            # Get old label full file path
            old_label_file_path = os.path.join(old_labels_folder, ball_label_file_name)
            with open(old_label_file_path, "a") as old_label_file:
                old_label_file.write(label_line)

def train_validation_split(source_folder: str):
    """Creates train, validation and test folders in the data folder.
    """
    # Source folder
    images_folder = os.path.join(source_folder, "images")
    labels_folder = os.path.join(source_folder, "labels")
    images = sorted(os.listdir(images_folder))
    labels = sorted(os.listdir(labels_folder))
    # Train setup
    images_train_folder = os.path.join(source_folder, "train", "images")
    labels_train_folder = os.path.join(source_folder, "train", "labels")
    num_train_frames = 3737
    # Validation setup
    images_valid_folder = os.path.join(source_folder, "valid", "images")
    labels_valid_folder = os.path.join(source_folder, "valid", "labels")
    num_valid_frames = 1068
    # Test setup
    images_test_folder = os.path.join(source_folder, "test", "images")
    labels_test_folder = os.path.join(source_folder, "test", "labels")
    num_test_frames = 534
    for ind in range(num_train_frames + num_valid_frames + num_test_frames):
        src_img_file = os.path.join(images_folder, images[ind])
        src_label_file = os.path.join(labels_folder, labels[ind])
        if ind < num_train_frames:
            dst_img_file = os.path.join(images_train_folder, images[ind])
            dst_label_file = os.path.join(labels_train_folder, labels[ind])
        elif ind >= num_train_frames and ind < num_train_frames + num_valid_frames:
            dst_img_file = os.path.join(images_valid_folder, images[ind])
            dst_label_file = os.path.join(labels_valid_folder, labels[ind])
        elif ind >= num_train_frames + num_valid_frames:
            dst_img_file = os.path.join(images_test_folder, images[ind])
            dst_label_file = os.path.join(labels_test_folder, labels[ind])
        # Copy image and label file
        shutil.copyfile(src_img_file, dst_img_file)
        shutil.copyfile(src_label_file, dst_label_file)
        
        
if __name__ == "__main__":
    # merge("data/labelimg_import", "data/roboflow_football_import/labels")
    train_validation_split("data/roboflow_football_import")