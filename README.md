# Football player tracking using YOLOv5 :soccer:

![T7 detections](https://github.com/as51340/FootballPlayerTracking/blob/master/images/t7_detections.png?raw=true)

## Introduction

Player tracking is a very challenging problem for which some solutions already exist in the industry. It requires a complex solution based on deep learning so it can be automated without the need for human intervention. From such a solution, people fulfilling roles of football coaches, analysts and managers could benefit as well as clubs because of the possible cost reduction. The greatest benefit out of a player tracking system could get small clubs not having enough resources for hiring scouts and specialized analysts.

## Problem description

In this project we wanted to test capabilities of the YOLOv5 object detection system together with some of the compatible object trackers like OSNet and StrongSORT. The main goal was to check the performance of the object tracking solution when fine-tuned on the appropriate data coming from football matches and compare it with the basic pre-trained YOLOv5 model’s performance. The simple but plausible solution for classifying teams based on the image masking is provided too since it can be a nice feature in the analysis systems. The whole solution has been integrated with the ClearML system for logging purposes and with the Roboflow system for easier dataset management. For the purpose of integration with the Roboflow system, the script for converting Mot16 into Yolov5 torch format was also needed. Since the YOLOv5 is quite a complex solution, the theoretical explanation of AutoAnchor and Non-max suppression algorithm is provided, together with a chapter describing how the data could be managed in the parallel environment and explaining how YOLOv5 developers made it in an efficient manner .

## Dataset

Unfortunately, the dataset isn't public so it cannot be downloaded. It consists mainly of few matches recordings from 1st Croatian national league. The main data comes from a 5-minute clip and consists of 7500 frames. A part of this dataset was used for fine-tuning the pre-trained YOLOv5 model and the rest was left for validation purposes. After splitting the dataset locally, the Roboflow system was used to persist the data as well as to provide an easy dataset download scheme.

## Results

As it is mentioned above, several approaches were tried. First, a pre-trained YOLO models were tested to check their performance. This is visualized in the bar plot below:

![Pretrained models visualization](https://github.com/as51340/FootballPlayerTracking/blob/master/images/pretrained_models_comparison.png?raw=true)

After performing various experiments with pre-trained models, the time came to fine-tune the model on a part of the data from t7 data. Let’s take a moment here and explain what has actually been done. Object tracking consists of an object detector that is glued with an object tracker. Fine-tuning model actually means training an object detector while a tracker is used as it is given. We performed two experiments. In both experiments a batch of size 4 was used because of the hardware restrictions and large image sizes. In the first experiment, a model was trained for 100 epochs, while the second time model was trained for a total 300 epochs. Model’s performance is visualized on the plots below and summarized in table 5.

![Training performance](https://github.com/as51340/FootballPlayerTracking/blob/master/images/training_performance.png?raw=true)

![Results](https://github.com/as51340/FootballPlayerTracking/blob/master/images/results.png?raw=true)

## Problems

Several problems were noticed during the process of an object detection. When two players are one behind another, it is impossible for the object detector to detect both of them.

![Object occlusion](https://github.com/as51340/FootballPlayerTracking/blob/master/images/object_occlusion.png?raw=true)

It is interesting that the model successfully learnt not to classify assistant referees. However, in every frame, the main referee is classified as a player and this should be fixed in the future. It was also noticed that sometimes when players come close enough to each other, the tracker switches their id so it fails to continue providing correct trajectories.


![Switched ID 1](https://github.com/as51340/FootballPlayerTracking/blob/master/images/switched_id_one.png?raw=true)

![Switched ID 2](https://github.com/as51340/FootballPlayerTracking/blob/master/images/switched_id_two.png?raw=true)

The issue that most object trackers have when dealing with small objects is to assign one object always the same id. For example, in such a use case it would be very useful to have one trajectory for one player for the whole game. However, what happens is that sometimes the object tracker says there is a new player on the scene and assigns him a new id whereas in the match, there are always only 22 players.


![Reid one](https://github.com/as51340/FootballPlayerTracking/blob/master/images/reid_one.png?raw=true)

![Reid two](https://github.com/as51340/FootballPlayerTracking/blob/master/images/reid_two.png?raw=true)

The big question is whether the model which is trained on only a part of one match, can be reused for detection on other matches. On the t16 data the model works really well although images from the new match aren’t of the same size as those from the training dataset. However, occasionally it fails to detect some players which can be seen in the image below. On the t16 data, the model also correctly chooses not to detect assistant referees and coaches near the pitch’s boundaries.

![Detection missed](https://github.com/as51340/FootballPlayerTracking/blob/master/images/missed_detection.png?raw=true)

On the t5 data, the model detects persons which are not players in the field. Image 18 shows one such output in which the 4th referee, together with security people, players on the bench and players that are warming up are classified together with players on the pitch. A possible reason could be the different stadium on which this match was played and hence the camera captures a different perspective compared to the stadium from the first two matches. A possible solution could be to create a perspective transformation of the whole scene to the bird’s eye and to detect only objects that are within pitch’s boundaries.

![t5 dataset](https://github.com/as51340/FootballPlayerTracking/blob/master/images/t5_detections.png?raw=true)

## Hardware requirements

YOLOv5 is a PyTorch model that makes a great use of the hardware architecture. There are several things YOLOv5 developers did so the code could be as performant as possible. YOLOv5 is considered a very fast and efficient algorithm and one of the things that makes it fast is the caching algorithm. No literature was found on that topic, so conclusions are made based on a debugging session made as a part of this project. Cache file ends in .cache, e.g the path could look like following /home/andi/FER/year5/PlayerTracking/datasets/coco128/labels/train2017.cache.
First it needs to be checked whether the .cache file exists. The cache files store information about each image = path, labels and shapes in the pickle format. From labels, label_files can be obtained where in each file, information about ground true detections have been stored in the YOLOv5 format. One line in such a file corresponds to one detection and in each line there is information about class id, center (x, y) coordinates, width and height of the bounding box. Images can be cached on the disk or in the RAM. In both cases ThreadPool is used to control a pool of worker threads to which jobs are submitted. If the images are cached to the disk, then they are saved as .npy files; if not, all images just take a part of the RAM.
To be as efficient as possible, YOLOv5 developers also made use of torch’s distributed framework that allows training on multiple GPUs on a single or on a multi machine. This is an example of the single-program multiple-data training paradigm in which every model replica contains a different set of input data samples and the framework itself takes care of synchronizing replicas and overlapping gradient computation by averaging gradients from each node. There are several things one should be aware of when trying to make use of the GPU training. The first goal is always to minimize the amount of data transferred between the host and the device, where the device in this case is GPU and the host is CPU. However, data transfer is in some cases necessary so when we have to do it, we have to seek for transferring as much data as possible in order to eliminate per-transfer overhead. A great way to minimize a communication time is to interleave it with a computation. This can be in the case of the GPU, multiple kernels execution.
Transferring the data directly from the host to the CUDA device is impossible on modern computers. This is because the device requires the memory to be page-locked (unpageable). Today’s computer memory management is based on the concept called virtual memory. Every program itself has its own memory so it reads and writes to the entity called pages. Each page is then mapped to the secondary storage via a page table. What this enables us is better memory utilization because each program sees consecutive logical blocks of memory which are then mapped to the different parts in e.g secondary storage. However, the data cannot be directly transferred from the paged memory to the CUDA device which is why a page-locked (pinnable) memory must be reserved on the host side. This incurs significant costs because we have to transfer the data twice, the first time when sending the data from paged memory to the pinnable memory on the host and the second time from pinnable memory to the CUDA device. Torch framework allows us to have memory pinning for data buffers and it is encouraged to use it when it is known that some time will be transferred multiple times to the CUDA device.


## Conclusion

Experiments showed that fine-tuning significantly improves a model's performance when compared to the pre-trained model. It is shown that the environment in which the match is played (pitch) is very important for the model’s performance because it directly influences things seen with the camera. When object occlusion occurs, the YOLOv5 model fails to detect both objects. As in many other player tracking solutions, person reidentification is needed so the same person doesn’t have several ID labels during the tracking process. A solution is proposed for solving unwanted player detections outside the pitch.
The last chapter describes how the YOLOv5 model makes use of a parallel environment and provides a few useful techniques for training models in a distributed setting.
Results could be further improved by training a model for a longer time, having larger batch size and searching for optimal hyperparameters.














