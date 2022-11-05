import numpy as np
import cv2

colors = {"red": ([17, 15, 75], [50, 56, 200]), "blue": ([43, 31, 4], [250, 88, 50]), "white": ([187,169,112],[255,255,255])}
          
          # "green": ([45, 100, 20], [55, 255,255]), "yellow": ([25, 100, 20], [35, 255, 255])}

def count_nonblack_np(img):
    return img.any(axis=-1).sum()

def color_detection(image, show = False) -> str:
    
    max_ratio, max_color = None, None
    
    for color, (lower, upper) in colors.items():
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)
        tot_pix = count_nonblack_np(image)
        color_pix = count_nonblack_np(output)
        ratio = color_pix/tot_pix
        
        if max_ratio is None or ratio > max_ratio:
            max_ratio, max_color = ratio, color
      
    return max_color
    
def green_range():
    greenBGR = np.uint8([[[0,255,0 ]]])
    hsv_green = cv2.cvtColor(greenBGR, cv2.COLOR_BGR2HSV)
    print(hsv_green)
