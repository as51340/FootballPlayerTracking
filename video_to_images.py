# ---------------------------------------------------------------- #
# Reads a video and saves the frames as images in hardcoded folder #
# ---------------------------------------------------------------- #

import cv2

# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture("data/t7.mp4")

if vid_capture.isOpened() == False:
    print("Error opening the video file")
# Read fps and frame count
else:
    # Get frame rate information
    # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    fps = vid_capture.get(5)
    print("Frames per second : ", fps, "FPS")

    # Get frame count
    # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    frame_count = vid_capture.get(7)
    print("Frame count : ", frame_count)

i = 0
while vid_capture.isOpened():
    ret, frame = vid_capture.read()
    if ret == True:
        if i % 100 == 0:
            print(f"Frame: {i}")
        cv2.imwrite(
            f"data/t7/roboflow_football_import/images/frame{str(i).zfill(4)}.jpg", frame
        )

        # 20 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(20)

        if key == ord("q"):
            break
    else:
        break
    i += 1

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()
