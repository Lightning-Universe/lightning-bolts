# YOLO

The YOLO model has evolved quite a bit, since the original publication in 2016. The original source code was written in C, using a framework called [Darknet](https://github.com/pjreddie/darknet). The final revision by the original author was called YOLOv3 and described in an [arXiv paper](https://arxiv.org/abs/1804.02767). Later various other authors have written implementations that improve different aspects of the model or the training procedure. [YOLOv4 implementation](https://github.com/AlexeyAB/darknet) was still based on Darknet and [YOLOv5](https://github.com/ultralytics/yolov5) was written using PyTorch. Most other implementations are based on these.

This PyTorch Lightning implementation combines features from some of the notable YOLO implementations. The most important papers are:

- *YOLOv3*: [Joseph Redmon and Ali Farhadi](https://arxiv.org/abs/1804.02767)
- *YOLOv4*: [Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao](https://arxiv.org/abs/2004.10934)
- *YOLOv7*: [Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao](https://arxiv.org/abs/2207.02696)
- *Scaled-YOLOv4*: [Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao](https://arxiv.org/abs/2011.08036)
- *YOLOX*: [Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun](https://arxiv.org/abs/2107.08430)

## Network Architecture

Any network can be used with YOLO detection heads as long as it produces feature maps with the correct number of features. Typically the network consists of a CNN backbone combined with a [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) or a [Path Aggregation Network](https://arxiv.org/abs/1803.01534). Backbone layers reduce the size of the feature map and the network may contain multiple detection heads that operate at different resolutions.

The user can write the network architecture in PyTorch, or construct a computational graph based on a Darknet configuration file using the [`DarknetNetwork`](https://github.com/Lightning-AI/lightning-bolts/tree/master/pl_bolts/models/detection/yolo/darknet_network.py) class. The network object is passed to the YOLO constructor in the `network` argument. `DarknetNetwork` is also able to read weights from a Darknet model file.

There are several network architectures included in the [`torch_networks`](https://github.com/Lightning-AI/lightning-bolts/tree/master/pl_bolts/models/detection/yolo/torch_networks.py) module (YOLOv4, YOLOv5, YOLOX). Larger and smaller variants of these models can be created by varying the `width` and `depth` arguments.

## Anchors

A detection head can try to detect objects at each of the anchor points that are spaced evenly across the image in a grid. The size of the grid is determined by the width and height of the feature map. There can be a number of anchors (typically three) per grid cell. The number of features predicted per grid cell has to be `(5 + num_classes) * anchors_per_cell`.

The width and the height of a bounding box is detected relative to a prior shape. `anchors_per_cell` prior shapes per detection head are defined in the network configuration. That is, if the network uses three detection heads, and each head detects three bounding boxes per grid cell, nine prior shapes need to be defined. They are defined in the Darknet configuration file or provided to the network class constructor. The default values have been obtained by clustering bounding box shapes in the COCO dataset. Note that if you use a different image size, you probably want to scale the prior shapes too.

The prior shapes are also used for matching the ground-truth targets to anchors during training. With the exception of the SimOTA matching algorithm, targets are matched only to anchors from the closest grid cell. The prior shapes are used to determine, to which anchors from that cell the target is matched. The losses are computed between the targets boxes and the predictions that correspond to their matched anchors. Different matching rules have been implemented:

- *maxiou*: The original matching rule that matches a target to the prior shape that gives the highest IoU.
- *iou*: Matches a target to an anchor, if the IoU between the target and the prior shape is above a threshold. Multiple anchors may be matched to the same target, and the loss will be computed from a number of pairs that is generally not the same as the number of ground-truth boxes.
- *size*: Calculates the ratio between the width and height of the target box to the prior width and height. If both the width and the height are close enough to the prior shape, matches the target to the anchor.
- *simota*: The SimOTA matching algorithm from YOLOX. Targets can be matched not only to anchors from the closest grid cell, but to any anchors that are inside the target bounding box and whose prior shape is close enough to the target shape. The matching algorithm is based on Optimal Transport and uses the training loss between the target and the predictions as the cost. That is, the prior shapes are not used for matching, but the predictions corresponding to the anchors.

## Input Data

The model input is expected to be a list of images. Each image is a tensor with shape `[channels, height, width]`. The images from a single batch will be stacked into a single tensor, so the sizes have to match. Different batches can have different image sizes. The feature pyramid network introduces another constraint on the image size: the width and the height have to be divisible by the ratio in which the network downsamples the input.

During training, the model expects both the image tensors and a list of targets. It's possible to train a model using one integer class label per box, but the YOLO model supports also multiple labels per box. For multi-label training, simply use a boolean matrix that indicates which classes are assigned to which boxes, in place of the class labels. Each target is a dictionary containing the following tensors:

- *boxes*: `(x1, y1, x2, y2)` coordinates of the ground-truth boxes in a matrix with shape `[N, 4]`.
- *labels*: Either integer class labels in a vector of size `N` or a class mask for each ground-truth box in a boolean matrix with shape `[N, classes]`

## Training

The command line application demonstrates how to train a YOLO model using PyTorch Lightning. The first step is to create a network, either from a Darknet configuration file, or using one of the included PyTorch networks. The network is passed to the YOLO model constructor.

The data module needs to resize the data to a suitable size, in addition to any augmenting transforms. For example, YOLOv4 network requires that the width and the height are multiples of 32.

## Inference

During inference, the model requires only the input images. `forward()` method receives a mini-batch of images in a tensor with shape `[N, channels, height, width]`.

Every detection head predicts a bounding box at every anchor. `forward()` returns the predictions from all detection heads in a tensor with shape `[N, anchors, classes + 5]`, where `anchors` is the total number of anchors in all detection heads. The predictions are `x1`, `y1`, `x2`, `y2`, confidence, and the probability for each class. The coordinates are scaled to the input image size.

`infer()` method filters and processes the predictions. A class-specific score is obtained by multiplying the class probability with the detection confidence. Only detections with a high enough score are kept. YOLO does not use `softmax` to normalize the class probabilities, but each probability is normalized individually using `sigmoid`. Consequently, one object can be assigned to multiple categories. If more than one class has a score that is above the confidence threshold, these will be split into multiple detections during postprocessing. Then the detections are filtered using non-maximum suppression. The processed output is returned in a dictionary containing the following tensors:

- *boxes*: a matrix of predicted bounding box `(x1, y1, x2, y2)` coordinates in image space
- *scores*: a vector of detection confidences
- *labels*: a vector of predicted class labels
