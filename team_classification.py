import numpy as np
import cv2

colors = {"red": ([17, 15, 75], [50, 56, 200]), "blue": ([43, 31, 4], [250, 88, 50]), "white": ([187,169,112],[255,255,255]), "black": ([0, 0, 0], [255, 255, 60])}

def count_nonblack_np(img):
    return img.any(axis=-1).sum()

def color_detection(image, show = False) -> str:
    for color, (lower, upper) in colors.items():
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)
        tot_pix = count_nonblack_np(image)
        color_pix = count_nonblack_np(output)
        ratio = color_pix/tot_pix
        
        if ratio > 0.01:
            print(color)
            return color
        
    print(f"Color not identified")
    return "white"