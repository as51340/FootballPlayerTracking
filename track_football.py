import argparse

import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import matplotlib.pyplot as plt

from team_classification import color_detection
from match import Match, Player, Team


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / "weights"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / "yolov5") not in sys.path:
    sys.path.append(str(ROOT / "yolov5"))  # add yolov5 ROOT to PATH
if str(ROOT / "trackers" / "strong_sort") not in sys.path:
    sys.path.append(
        str(ROOT / "trackers" / "strong_sort")
    )  # add strong_sort ROOT to PATH
if str(ROOT / "trackers" / "ocsort") not in sys.path:
    sys.path.append(str(ROOT / "trackers" / "ocsort"))  # add strong_sort ROOT to PATH
if (
    str(ROOT / "trackers" / "strong_sort" / "deep" / "reid" / "torchreid")
    not in sys.path
):
    sys.path.append(
        str(ROOT / "trackers" / "strong_sort" / "deep" / "reid" / "torchreid")
    )  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (
    LOGGER,
    check_img_size,
    non_max_suppression,
    scale_coords,
    check_requirements,
    cv2,
    check_imshow,
    xyxy2xywh,
    increment_path,
    strip_optimizer,
    colorstr,
    print_args,
    check_file,
)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

# Constants to keep tracking parameters more easily
MODEL = "model"
ENSEMBLE = "ensemble"
WARMUP = "warmup"
REFEREE = "referee"
CROPS = "crops"
FRAME = "frame"
TRACKS = "tracks"
REFERENCE_IMG_PATH = "reference_img.jpg"
JPG = ".jpg"
TXT = ".txt"
PLAYER_CLASS = 0
PLAYER = "player"
BALL_CLASS = 32
BALL = "ball"
YOLO_WEIGHTS_DEFAULT = "yolov5m.pt"
APPEARENCE_DESCRIPTOR_WEIGHTS_DEFAULT = "osnet_x0_25_msmt17.pt"
STRONGSORT = "strongsort"
PROJECT_FOLDER_DEF = "runs/track"


def get_exp_name(yolo_weights, name):
    """
    Just returns exp name.
    """
    # Check instance of yolo_weights
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif (
        type(yolo_weights) is list and len(yolo_weights) == 1
    ):  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = ENSEMBLE
    return name if name else exp_name + "_" + strong_sort_weights.stem

def directories(exp_name, project, exist_ok, save_txt):
    """
    Creates directory if needed.
    """
    save_dir = increment_path(
        Path(project) / exp_name, exist_ok=exist_ok
    )  # increment run
    (save_dir / TRACKS if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir
    return save_dir

def get_device(eval, device):
    """
    Loads manually device if the eval flag is specified or uses the select_device method for automatically doing stuff.
    """
    if eval:
        return torch.device(int(device))
    else:
        return select_device(device)
    
def load_trackers(nr_sources, tracking_method, appearance_descriptor_weights, device, half):
    """
    Returns list of trackers
    """
    tracker_list = []
    for i in range(nr_sources):
        # Create and append tracker
        tracker = create_tracker(tracking_method, appearance_descriptor_weights, device, half)
        tracker_list.append(tracker)
        # Warmup model if needed
        if hasattr(tracker_list[i], MODEL):
            if hasattr(tracker_list[i].model, WARMUP):
                tracker_list[i].model.warmup()
    return tracker_list


def save_detections_MOT_txt(txt_path, frame_idx, id, bbox_left, bbox_top, bbox_w, bbox_h, i):
    # Write MOT compliant results to file
    with open(txt_path + TXT, "a") as f:  # open file on the txt path for the reading
        f.write(
            ("%g " * 10 + "\n") # %g = Floating point format. Uses lowercase exponential format if exponent is less than -4 or not less than precision, decimal format otherwise.
            % (
                frame_idx + 1,
                id,
                bbox_left,
                bbox_top,
                bbox_w,
                bbox_h,
                -1,
                -1,
                -1,
                i,
            )
        )

def save_video_with_detections(vid_path, save_path, vid_writer, vid_cap, im0, i):
    """
    This method is called for every detection in one image.
    """
    if vid_path[i] != save_path:  # new video
        vid_path[i] = save_path
        if isinstance(vid_writer[i], cv2.VideoWriter):
            vid_writer[i].release()  # release previous video writer
        if vid_cap:  # video
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:  # stream
            fps, w, h = 30, im0.shape[1], im0.shape[0]
        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    vid_writer[i].write(im0)
    
    
def get_txt_file_name_and_save_path(source, p, save_dir):
    """
    Returns txt file name and save path
    """
    # video file
    if source.endswith(VID_FORMATS):
        txt_file_name = p.stem
        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
    # folder with imgs
    else:
        txt_file_name = p.parent.name  # get folder name containing current img
        save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
        
    return txt_file_name, save_path
    
        
                                 
@torch.no_grad()
def run(
    source="0",
    yolo_weights=WEIGHTS / YOLO_WEIGHTS_DEFAULT,  # model.pt path(s),
    appearance_descriptor_weights=WEIGHTS / APPEARENCE_DESCRIPTOR_WEIGHTS_DEFAULT,  # model.pt path,
    tracking_method=STRONGSORT,
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    show_vid=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    save_vid=False,  # save confidences in --save-txt labels
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / PROJECT_FOLDER_DEF,  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=2,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    hide_class=False,  # hide IDs
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    eval=False,  # run multi-gpu eval
    max_frames=None,  # don't run whole video
    referee_color=None,  # color of the referee
    team1_color=None,  # color of the first team
    team2_color=None,  # color of the second team
):
    # Just process source
    source = str(source)

    # Create directories for saving results if needed
    exp_name = get_exp_name(yolo_weights, name)
    save_dir = directories(exp_name, project, exist_ok, save_txt)
    
    # Get device
    device = get_device(eval, device)
    
    # Load the model and the dataset
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    nr_sources = 1
    vid_path, vid_writer, txt_path, outputs = (
        [None] * nr_sources,
        [None] * nr_sources,
        [None] * nr_sources,
        [None] * nr_sources
    )
    # Create as many strong sort instances as there are video sources and warmup each of the trackers
    tracker_list = load_trackers(nr_sources, tracking_method, appearance_descriptor_weights, device, half)
    # Warmup the model
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))
    
    # Football tracking
    model.names[PLAYER_CLASS] = PLAYER
    model.names[BALL_CLASS] = BALL
    team_classification = team1_color and team2_color and referee_color
    if team_classification:
        match = Match()
        match.team1 = Team(color=team1_color)
        match.team2 = Team(color=team2_color)
        LOGGER.info("Running team classification...")
    else:
        LOGGER.info("Team classification not running...")
    
    # Run tracking
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()

        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Stop while in the development
        if max_frames != None and frame_idx == max_frames:
            break
        # Obtain the reference image for creating perspective transformation    
        if frame_idx == 0:
            cv2.imwrite(REFERENCE_IMG_PATH, im0s)
        
        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, FRAME, 0)
            p = Path(p)  # to Path
            txt_file_name, save_path = get_txt_file_name_and_save_path(source, p, save_dir)

            curr_frames[i] = im0

            txt_path = str(save_dir / TRACKS / txt_file_name)  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            # Create annotator
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            
            # If there are some detections
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                # Some very weird format TODO
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = tracker_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        # Get bounding boxes
                        bboxes = [int(i) for i in output[0:4]]
                        bbox_left = bboxes[0]
                        bbox_top = bboxes[1]
                        bbox_w = bboxes[2] - bboxes[0]
                        bbox_h = bboxes[3] - bboxes[1]
                        
                        # Get id of the tracked objects and the class
                        id = int(output[4])
                        c = int(output[5])

                        # Let's try to determine the color of each player
                        # First extract crops
                        txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ""
                        
                        # Get the crop and optionally save it
                        crop = save_one_box(
                            bboxes,
                            imc,
                            file=save_dir / CROPS / txt_file_name / names[c] / f"{id}" / f"{p.stem}{JPG}",
                            BGR=True,
                            save=save_crop,
                        )
                        
                        # Get the label, now if we are running team_classificication, that means we have to further separate class into team1, team2 and referee
                        label = names[c]
                        if c == PLAYER_CLASS and team_classification:
                            color = color_detection(crop)
                            label_flag = 1
                            if color == referee_color:
                                label_flag = 0
                                label = REFEREE
                            elif color == team1_color:
                                match.team1.add_player(Player(id=id))
                            elif color == team2_color:
                                label_flag = 2
                                match.team2.add_player(Player(id=id))
                            # If we are running team classification then do some custom processing for players and the referee
                            annotator.box_label(bboxes, label, color=colors(label_flag, True))
                        else:
                            # Otherwise just color everything in the same color
                            annotator.box_label(bboxes, label, color=colors(c, True))
                                
                        #  Save to the txt file frame with detections
                        if save_txt:
                            save_detections_MOT_txt(txt_path, frame_idx, id, bbox_left, bbox_top, bbox_w, bbox_h, i)

                LOGGER.info(f"{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)")

            else:
                # strongsort_list[i].increment_ages()
                LOGGER.info("No detections")

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)  # here the image is being shown
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                save_video_with_detections(vid_path, save_path, vid_writer, vid_cap, im0, i)
            
            # Get previous frame
            prev_frames[i] = curr_frames[i]
            
            # reset teams
            if team_classification:
                match.reset()

    # Print results
    t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}" % t)
    
    # Output that something is being savged
    if save_txt or save_vid:
        s = (
            f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}"
            if save_txt
            else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # If something needs update
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo-weights",
        nargs="+",
        type=Path,
        default=WEIGHTS / "yolov5m.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--appearance-descriptor-weights",
        type=Path,
        default=WEIGHTS / "osnet_x0_25_msmt17.pt",
    )
    parser.add_argument(
        "--tracking-method", type=str, default="strongsort", help="strongsort, ocsort"
    )
    parser.add_argument(
        "--source", type=str, default="0", help="file/dir/URL/glob, 0 for webcam"
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.5, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.5, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--show-vid", action="store_true", help="display tracking video results"
    )
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument(
        "--save-vid", action="store_true", help="save video tracking results"
    )
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default=ROOT / "runs/track", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=2, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--hide-class", default=False, action="store_true", help="hide IDs"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    parser.add_argument("--eval", action="store_true", help="run evaluation")
    parser.add_argument(
        "--max-frames", type=int, help="Run as many frames as you want. "
    )
    parser.add_argument(
        "--referee-color", type=str, help="Specify the color of the referee"
    )
    parser.add_argument(
        "--team1-color", type=str, help="Specify the color of the first team(home)"
    )
    parser.add_argument(
        "--team2-color", type=str, help="Specify the color of the second team(away)"
    )
 
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(
        requirements=ROOT / "requirements.txt", exclude=("tensorboard", "thop")
    )
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

