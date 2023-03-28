import sys

import numpy as np


def convert_file(path_to_file, path_to_save):
    detections = []
    with open(path_to_file, "r") as orig_file:
        while (line := orig_file.readline()):
            elements = line.strip().split(" ")
            elements = [float(elem) for elem in elements]
            detections.append(elements)
    np.save(path_to_save, np.array(detections, dtype=np.float32))


if __name__ == "__main__":
    path_to_file, path_to_save = sys.argv[1], sys.argv[2]
    convert_file(path_to_file, path_to_save)