# ----------------------------------------------------------------------------- #
# Performs conversion from MOT16 format to yolov5 pytorch format.               #
# ----------------------------------------------------------------------------- #

import cv2

FRAME_WIDTH = 3
FRAME_HEIHGT = 4
PLAYER_CLASS = 0

def get_video_shape(path):
    vcap = cv2.VideoCapture(path)  # 0=camera
    if vcap.isOpened():
        width = vcap.get(3)  # float `width`
        height = vcap.get(4)  # float `height
        return width, height
    else:
        raise Exception("Video cannot be opened")

def convert(video_path, mot16_file_path, roboflow_labels_dir):
    img_width, img_height = get_video_shape(video_path)
    print(f"Img width: {img_width} img height: {img_height}")
    with open(mot16_file_path, "r") as mot16:
        mot16_lines = mot16.readlines()
        for line in mot16_lines:
            yolov5_data = []
            mot16_data = line.split(",")
            # Perform all computation needed for yolov5 pytorch format
            yolov5_data.append(PLAYER_CLASS)
            bbox_left, bbox_top, bbox_width, bbox_height = mot16_data[2:6]
            bbox_left, bbox_top, bbox_width, bbox_height = (
                float(bbox_left),
                float(bbox_top),
                float(bbox_width),
                float(bbox_height),
            )
            center_x = (bbox_left + 0.5 * bbox_width) / img_width
            center_y = (bbox_top + 0.5 * bbox_height) / img_height
            wid = bbox_width / img_width
            hig = bbox_height / img_height
            yolov5_data.append(center_x)
            yolov5_data.append(center_y)
            yolov5_data.append(wid)
            yolov5_data.append(hig)
            # Get the frame id and write
            frame_id = str(int(mot16_data[0]) - 1)
            file_name = f"{roboflow_labels_dir}/frame{frame_id.zfill(4)}.txt"
            with open(file_name, "a") as yolov5_frame:
                yolov5_frame.write(" ".join([str(item) for item in yolov5_data]) + "\n")
            if int(frame_id) % 100 == 0:
                print(f"Processed frame id: {frame_id}")


if __name__ == "__main__":
    convert(
        "data/t7/t7.mp4",
        "data/t7/t7_GT.txt",
        "data/t7/roboflow_football_import/labels",
    )
