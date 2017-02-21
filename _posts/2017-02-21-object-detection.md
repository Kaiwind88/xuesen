---
layout: post
title:  "Object Detection"
date:   2017-02-21
desc: "Deep Learning for Object Detection"
keywords: "DL,Detection"
categories: [Deeplearning]
tags: [DL,Detection]
icon: icon-html
---

Papers
=========

## **Deep Neural Networks for Object Detection**

- paper: [http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

## **OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1312.6229](http://arxiv.org/abs/1312.6229)
- github: [https://github.com/sermanet/OverFeat](https://github.com/sermanet/OverFeat)
- code: [http://cilvr.nyu.edu/doku.php?id=software:overfeat:start](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start)

# [R-CNN]()

## **Rich feature hierarchies for accurate object detection and semantic segmentation**

- intro: R-CNN
- arxiv: [http://arxiv.org/abs/1311.2524](http://arxiv.org/abs/1311.2524)
- supp: [http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf](http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf)
- slides: [http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf](http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf)
- slides: [http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf)
- github: [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)
- notes: [http://zhangliliang.com/2014/07/23/paper-note-rcnn/](http://zhangliliang.com/2014/07/23/paper-note-rcnn/)
- caffe-pr("Make R-CNN the Caffe detection example"): [https://github.com/BVLC/caffe/pull/482](https://github.com/BVLC/caffe/pull/482) 

# [MultiBox]()

## **Scalable Object Detection using Deep Neural Networks**

- intro: first MultiBox. Train a CNN to predict Region of Interest.
- arxiv: [http://arxiv.org/abs/1312.2249](http://arxiv.org/abs/1312.2249)
- github: [https://github.com/google/multibox](https://github.com/google/multibox)
- blog: [https://research.googleblog.com/2014/12/high-quality-object-detection-at-scale.html](https://research.googleblog.com/2014/12/high-quality-object-detection-at-scale.html)

## **Scalable, High-Quality Object Detection**

- intro: second MultiBox
- arxiv: [http://arxiv.org/abs/1412.1441](http://arxiv.org/abs/1412.1441)
- github: [https://github.com/google/multibox](https://github.com/google/multibox)

# [SPP-Net]()

## **Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**

- intro: ECCV 2014 / TPAMI 2015
- arxiv: [http://arxiv.org/abs/1406.4729](http://arxiv.org/abs/1406.4729)
- github: [https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)
- notes: [http://zhangliliang.com/2014/09/13/paper-note-sppnet/](http://zhangliliang.com/2014/09/13/paper-note-sppnet/)

# [DeepID-Net]()

## **DeepID-Net: Deformable Deep Convolutional Neural Networks for Object Detection**

- intro: PAMI 2016
- intro: an extension of R-CNN. box pre-training, cascade on region proposals, deformation layers and context representations
- project page: [http://www.ee.cuhk.edu.hk/%CB%9Cwlouyang/projects/imagenetDeepId/index.html](http://www.ee.cuhk.edu.hk/%CB%9Cwlouyang/projects/imagenetDeepId/index.html)
- arxiv: [http://arxiv.org/abs/1412.5661](http://arxiv.org/abs/1412.5661)

## **Object Detectors Emerge in Deep Scene CNNs**

- arxiv: [http://arxiv.org/abs/1412.6856](http://arxiv.org/abs/1412.6856)
- paper: [https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf](https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf)
- paper: [https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf](https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf)
- slides: [http://places.csail.mit.edu/slide_iclr2015.pdf](http://places.csail.mit.edu/slide_iclr2015.pdf)

## **segDeepM: Exploiting Segmentation and Context in Deep Neural Networks for Object Detection**

- intro: CVPR 2015
- project(code+data): [https://www.cs.toronto.edu/~yukun/segdeepm.html](https://www.cs.toronto.edu/~yukun/segdeepm.html)
- arxiv: [https://arxiv.org/abs/1502.04275](https://arxiv.org/abs/1502.04275)
- github: [https://github.com/YknZhu/segDeepM](https://github.com/YknZhu/segDeepM)

# [NoC]()

## **Object Detection Networks on Convolutional Feature Maps**

- intro: TPAMI 2015
- arxiv: [http://arxiv.org/abs/1504.06066](http://arxiv.org/abs/1504.06066)

## **Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction**

- arxiv: [http://arxiv.org/abs/1504.03293](http://arxiv.org/abs/1504.03293)
- slides: [http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf](http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf)
- github: [https://github.com/YutingZhang/fgs-obj](https://github.com/YutingZhang/fgs-obj)

# [Fast R-CNN]()

## **Fast R-CNN**

- arxiv: [http://arxiv.org/abs/1504.08083](http://arxiv.org/abs/1504.08083)
- slides: [http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf)
- github: [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
- webcam demo: [https://github.com/rbgirshick/fast-rcnn/pull/29](https://github.com/rbgirshick/fast-rcnn/pull/29)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/](http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/)
- notes: [http://blog.csdn.net/linj_m/article/details/48930179](http://blog.csdn.net/linj_m/article/details/48930179)
- github("Fast R-CNN in MXNet"): [https://github.com/precedenceguo/mx-rcnn](https://github.com/precedenceguo/mx-rcnn)
- github: [https://github.com/mahyarnajibi/fast-rcnn-torch](https://github.com/mahyarnajibi/fast-rcnn-torch)
- github: [https://github.com/apple2373/chainer-simple-fast-rnn](https://github.com/apple2373/chainer-simple-fast-rnn)
- github(Tensorflow): [https://github.com/zplizzi/tensorflow-fast-rcnn](https://github.com/zplizzi/tensorflow-fast-rcnn)

# [DeepBox]()

## **DeepBox: Learning Objectness with Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1505.02146](http://arxiv.org/abs/1505.02146)
- github: [https://github.com/weichengkuo/DeepBox](https://github.com/weichengkuo/DeepBox)

## **Object detection via a multi-region & semantic segmentation-aware CNN model**

- intro: ICCV 2015. MR-CNN
- arxiv: [http://arxiv.org/abs/1505.01749](http://arxiv.org/abs/1505.01749)
- github: [https://github.com/gidariss/mrcnn-object-detection](https://github.com/gidariss/mrcnn-object-detection)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/](http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/)
- notes: [http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/](http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/)
- my notes: Who can tell me why there are a bunch of duplicated sentences in section 7.2 "Detection error analysis"? :-D

# [Faster R-CNN]()

## **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**

- intro: NIPS 2015
- arxiv: [http://arxiv.org/abs/1506.01497](http://arxiv.org/abs/1506.01497)
- gitxiv: [http://www.gitxiv.com/posts/8pfpcvefDYn2gSgXk/faster-r-cnn-towards-real-time-object-detection-with-region](http://www.gitxiv.com/posts/8pfpcvefDYn2gSgXk/faster-r-cnn-towards-real-time-object-detection-with-region)
- slides: [http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf](http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf)
- github: [https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)
- github: [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- github: [https://github.com/mitmul/chainer-faster-rcnn](https://github.com/mitmul/chainer-faster-rcnn)
- github(Torch): [https://github.com/andreaskoepf/faster-rcnn.torch](https://github.com/andreaskoepf/faster-rcnn.torch)
- github(Torch): [https://github.com/ruotianluo/Faster-RCNN-Densecap-torch](https://github.com/ruotianluo/Faster-RCNN-Densecap-torch)
- github(Tensorflow): [https://github.com/smallcorgi/Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)
- github(Tensorflow): [https://github.com/CharlesShang/TFFRCNN](https://github.com/CharlesShang/TFFRCNN)

## **Faster R-CNN in MXNet with distributed implementation and data parallelization**

- github: [https://github.com/dmlc/mxnet/tree/master/example/rcnn](https://github.com/dmlc/mxnet/tree/master/example/rcnn)

## **Contextual Priming and Feedback for Faster R-CNN**

- intro: ECCV 2016. Carnegie Mellon University
- paper: [http://abhinavsh.info/context_priming_feedback.pdf](http://abhinavsh.info/context_priming_feedback.pdf)
- poster: [http://www.eccv2016.org/files/posters/P-1A-20.pdf](http://www.eccv2016.org/files/posters/P-1A-20.pdf)

## **An Implementation of Faster RCNN with Study for Region Sampling**

- intro: Technical Report, 3 pages. CMU
- arxiv: [https://arxiv.org/abs/1702.02138](https://arxiv.org/abs/1702.02138)
- github: [https://github.com/endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)

# [YOLO]()

## **You Only Look Once: Unified, Real-Time Object Detection**

![](https://camo.githubusercontent.com/e69d4118b20a42de4e23b9549f9a6ec6dbbb0814/687474703a2f2f706a7265646469652e636f6d2f6d656469612f66696c65732f6461726b6e65742d626c61636b2d736d616c6c2e706e67)

- arxiv: [http://arxiv.org/abs/1506.02640](http://arxiv.org/abs/1506.02640)
- code: [http://pjreddie.com/darknet/yolo/](http://pjreddie.com/darknet/yolo/)
- github: [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/](https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/)
- github: [https://github.com/gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)
- github: [https://github.com/xingwangsfu/caffe-yolo](https://github.com/xingwangsfu/caffe-yolo)
- github: [https://github.com/frankzhangrui/Darknet-Yolo](https://github.com/frankzhangrui/Darknet-Yolo)
- github: [https://github.com/BriSkyHekun/py-darknet-yolo](https://github.com/BriSkyHekun/py-darknet-yolo)
- github: [https://github.com/tommy-qichang/yolo.torch](https://github.com/tommy-qichang/yolo.torch)
- github: [https://github.com/frischzenger/yolo-windows](https://github.com/frischzenger/yolo-windows)
- gtihub: [https://github.com/AlexeyAB/yolo-windows](https://github.com/AlexeyAB/yolo-windows)

## **darkflow - translate darknet to tensorflow. Load trained weights, retrain/fine-tune them using tensorflow, export constant graph def to C++**

- blog: [https://thtrieu.github.io/notes/yolo-tensorflow-graph-buffer-cpp](https://thtrieu.github.io/notes/yolo-tensorflow-graph-buffer-cpp)
- github: [https://github.com/thtrieu/darkflow](https://github.com/thtrieu/darkflow)

## **Start Training YOLO with Our Own Data**

![](http://guanghan.info/blog/en/wp-content/uploads/2015/12/images-40.jpg)

- intro: train with customized data and class numbers/labels. Linux / Windows version for darknet.
- blog: [http://guanghan.info/blog/en/my-works/train-yolo/](http://guanghan.info/blog/en/my-works/train-yolo/)
- github: [https://github.com/Guanghan/darknet](https://github.com/Guanghan/darknet)

## **R-CNN minus R**

- arxiv: [http://arxiv.org/abs/1506.06981](http://arxiv.org/abs/1506.06981)

## **AttentionNet: Aggregating Weak Directions for Accurate Object Detection**

- intro: ICCV 2015
- intro: state-of-the-art performance of 65% (AP) on PASCAL VOC 2007/2012 human detection task
- arxiv: [http://arxiv.org/abs/1506.07704](http://arxiv.org/abs/1506.07704)
- slides: [https://www.robots.ox.ac.uk/~vgg/rg/slides/AttentionNet.pdf](https://www.robots.ox.ac.uk/~vgg/rg/slides/AttentionNet.pdf)
- slides: [http://image-net.org/challenges/talks/lunit-kaist-slide.pdf](http://image-net.org/challenges/talks/lunit-kaist-slide.pdf)

## **DenseBox: Unifying Landmark Localization with End to End Object Detection**

- arxiv: [http://arxiv.org/abs/1509.04874](http://arxiv.org/abs/1509.04874)
- demo: [http://pan.baidu.com/s/1mgoWWsS](http://pan.baidu.com/s/1mgoWWsS)
- KITTI result: [http://www.cvlibs.net/datasets/kitti/eval_object.php](http://www.cvlibs.net/datasets/kitti/eval_object.php)

## **SSD: Single Shot MultiBox Detector**

<div style="text-align: center">
<img src="https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67" width = "70%" height = "70%"/>
</div><br>
- intro: ECCV 2016 Oral
- arxiv: [http://arxiv.org/abs/1512.02325](http://arxiv.org/abs/1512.02325)
- paper: [http://www.cs.unc.edu/~wliu/papers/ssd.pdf](http://www.cs.unc.edu/~wliu/papers/ssd.pdf)
- slides: [http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf](http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf)
- github: [https://github.com/weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd)
- video: [http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973](http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973)
- github(MXNet): [https://github.com/zhreshold/mxnet-ssd](https://github.com/zhreshold/mxnet-ssd)
- github: [https://github.com/zhreshold/mxnet-ssd.cpp](https://github.com/zhreshold/mxnet-ssd.cpp)
- github(Keras): [https://github.com/rykov8/ssd_keras](https://github.com/rykov8/ssd_keras)


## **Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks**

- intro: "0.8s per image on a Titan X GPU (excluding proposal generation) without two-stage bounding-box regression
and 1.15s per image with it".
- arxiv: [http://arxiv.org/abs/1512.04143](http://arxiv.org/abs/1512.04143)
- slides: [http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf](http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf)
- coco-leaderboard: [http://mscoco.org/dataset/#detections-leaderboard](http://mscoco.org/dataset/#detections-leaderboard)

## **Adaptive Object Detection Using Adjacency and Zoom Prediction**

- intro: CVPR 2016. AZ-Net
- arxiv: [http://arxiv.org/abs/1512.07711](http://arxiv.org/abs/1512.07711)
- github: [https://github.com/luyongxi/az-net](https://github.com/luyongxi/az-net)
- youtube: [https://www.youtube.com/watch?v=YmFtuNwxaNM](https://www.youtube.com/watch?v=YmFtuNwxaNM)

## G-CNN

**G-CNN: an Iterative Grid Based Object Detector**

- arxiv: [http://arxiv.org/abs/1512.07729](http://arxiv.org/abs/1512.07729)

**Factors in Finetuning Deep Model for object detection**

**Factors in Finetuning Deep Model for Object Detection with Long-tail Distribution**

- intro: CVPR 2016.rank 3rd for provided data and 2nd for external data on ILSVRC 2015 object detection
- project page: [http://www.ee.cuhk.edu.hk/~wlouyang/projects/ImageNetFactors/CVPR16.html](http://www.ee.cuhk.edu.hk/~wlouyang/projects/ImageNetFactors/CVPR16.html)
- arxiv: [http://arxiv.org/abs/1601.05150](http://arxiv.org/abs/1601.05150)

**We don't need no bounding-boxes: Training object class detectors using only human verification**

- arxiv: [http://arxiv.org/abs/1602.08405](http://arxiv.org/abs/1602.08405)

## HyperNet

**HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection**

- arxiv: [http://arxiv.org/abs/1604.00600](http://arxiv.org/abs/1604.00600)

## MultiPathNet

**A MultiPath Network for Object Detection**

- intro: BMVC 2016. Facebook AI Research (FAIR)
- arxiv: [http://arxiv.org/abs/1604.02135](http://arxiv.org/abs/1604.02135)
- github: [https://github.com/facebookresearch/multipathnet](https://github.com/facebookresearch/multipathnet)

## CRAFT

**CRAFT Objects from Images**

- intro: CVPR 2016. Cascade Region-proposal-network And FasT-rcnn. an extension of Faster R-CNN
- project page: [http://byangderek.github.io/projects/craft.html](http://byangderek.github.io/projects/craft.html)
- arxiv: [https://arxiv.org/abs/1604.03239](https://arxiv.org/abs/1604.03239)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_CRAFT_Objects_From_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_CRAFT_Objects_From_CVPR_2016_paper.pdf)
- github: [https://github.com/byangderek/CRAFT](https://github.com/byangderek/CRAFT)

## OHEM

**Training Region-based Object Detectors with Online Hard Example Mining**

- intro: CVPR 2016 Oral. Online hard example mining (OHEM)
- arxiv: [http://arxiv.org/abs/1604.03540](http://arxiv.org/abs/1604.03540)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)
- github（Official）: [https://github.com/abhi2610/ohem](https://github.com/abhi2610/ohem)
- author page: [http://abhinav-shrivastava.info/](http://abhinav-shrivastava.info/)

**Track and Transfer: Watching Videos to Simulate Strong Human Supervision for Weakly-Supervised Object Detection**

- intro: CVPR 2016
- arxiv: [http://arxiv.org/abs/1604.05766](http://arxiv.org/abs/1604.05766)

**Exploit All the Layers: Fast and Accurate CNN Object Detector with Scale Dependent Pooling and Cascaded Rejection Classifiers**

- intro: scale-dependent pooling  (SDP), cascaded rejection clas-sifiers (CRC)
- paper: [http://www-personal.umich.edu/~wgchoi/SDP-CRC_camready.pdf](http://www-personal.umich.edu/~wgchoi/SDP-CRC_camready.pdf)

## R-FCN

**R-FCN: Object Detection via Region-based Fully Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1605.06409](http://arxiv.org/abs/1605.06409)
- github: [https://github.com/daijifeng001/R-FCN](https://github.com/daijifeng001/R-FCN)
- github: [https://github.com/Orpine/py-R-FCN](https://github.com/Orpine/py-R-FCN)

**Weakly supervised object detection using pseudo-strong labels**

- arxiv: [http://arxiv.org/abs/1607.04731](http://arxiv.org/abs/1607.04731)

**Recycle deep features for better object detection**

- arxiv: [http://arxiv.org/abs/1607.05066](http://arxiv.org/abs/1607.05066)

## MS-CNN

**A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection**

- intro: ECCV 2016
- intro: 640×480: 15 fps, 960×720: 8 fps
- arxiv: [http://arxiv.org/abs/1607.07155](http://arxiv.org/abs/1607.07155)
- github: [https://github.com/zhaoweicai/mscnn](https://github.com/zhaoweicai/mscnn)
- poster: [http://www.eccv2016.org/files/posters/P-2B-38.pdf](http://www.eccv2016.org/files/posters/P-2B-38.pdf)

**Multi-stage Object Detection with Group Recursive Learning**

- intro: VOC2007: 78.6%, VOC2012: 74.9%
- arxiv: [http://arxiv.org/abs/1608.05159](http://arxiv.org/abs/1608.05159)

**Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection**

- intro: WACV 2017. SubCNN
- arxiv: [http://arxiv.org/abs/1604.04693](http://arxiv.org/abs/1604.04693)
- github: [https://github.com/yuxng/SubCNN](https://github.com/yuxng/SubCNN)

## PVANET

**PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection**

- intro: "less channels with more layers", concatenated ReLU, Inception, and HyperNet, batch normalization, residual connections
- arxiv: [http://arxiv.org/abs/1608.08021](http://arxiv.org/abs/1608.08021)
- github: [https://github.com/sanghoon/pva-faster-rcnn](https://github.com/sanghoon/pva-faster-rcnn)
- leaderboard(PVANet 9.0): [http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4)

**PVANet: Lightweight Deep Neural Networks for Real-time Object Detection**

- intro: Presented at NIPS 2016 Workshop on Efficient Methods for Deep Neural Networks (EMDNN). 
Continuation of [arXiv:1608.08021](https://arxiv.org/abs/1608.08021)
- arxiv: [https://arxiv.org/abs/1611.08588](https://arxiv.org/abs/1611.08588)

## GBD-Net

**Gated Bi-directional CNN for Object Detection**

- intro: The Chinese University of Hong Kong & Sensetime Group Limited
- paper: [http://link.springer.com/chapter/10.1007/978-3-319-46478-7_22](http://link.springer.com/chapter/10.1007/978-3-319-46478-7_22)
- mirror: [https://pan.baidu.com/s/1dFohO7v](https://pan.baidu.com/s/1dFohO7v)

**Crafting GBD-Net for Object Detection**

- intro: winner of the ImageNet object detection challenge of 2016. CUImage and CUVideo
- intro: gated bi-directional CNN (GBD-Net)
- arxiv: [https://arxiv.org/abs/1610.02579](https://arxiv.org/abs/1610.02579)
- github: [https://github.com/craftGBD/craftGBD](https://github.com/craftGBD/craftGBD)

## StuffNet

**StuffNet: Using 'Stuff' to Improve Object Detection**

- arxiv: [https://arxiv.org/abs/1610.05861](https://arxiv.org/abs/1610.05861)

**Generalized Haar Filter based Deep Networks for Real-Time Object Detection in Traffic Scene**

- arxiv: [https://arxiv.org/abs/1610.09609](https://arxiv.org/abs/1610.09609)

**Hierarchical Object Detection with Deep Reinforcement Learning**

- intro: Deep Reinforcement Learning Workshop (NIPS 2016)
- project page: [https://imatge-upc.github.io/detection-2016-nipsws/](https://imatge-upc.github.io/detection-2016-nipsws/)
- arxiv: [https://arxiv.org/abs/1611.03718](https://arxiv.org/abs/1611.03718)
- slides: [http://www.slideshare.net/xavigiro/hierarchical-object-detection-with-deep-reinforcement-learning](http://www.slideshare.net/xavigiro/hierarchical-object-detection-with-deep-reinforcement-learning)
- github: [https://github.com/imatge-upc/detection-2016-nipsws](https://github.com/imatge-upc/detection-2016-nipsws)
- blog: [http://jorditorres.org/nips/](http://jorditorres.org/nips/)

**Learning to detect and localize many objects from few examples**

- arxiv: [https://arxiv.org/abs/1611.05664](https://arxiv.org/abs/1611.05664)

**Speed/accuracy trade-offs for modern convolutional object detectors**

- intro: Google Research
- arxiv: [https://arxiv.org/abs/1611.10012](https://arxiv.org/abs/1611.10012)

**SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving**

- arxiv: [https://arxiv.org/abs/1612.01051](https://arxiv.org/abs/1612.01051)
- github: [https://github.com/BichenWuUCB/squeezeDet](https://github.com/BichenWuUCB/squeezeDet)

## Feature Pyramid Network (FPN)

**Feature Pyramid Networks for Object Detection**

- intro: Facebook AI Research
- arxiv: [https://arxiv.org/abs/1612.03144](https://arxiv.org/abs/1612.03144)

**Action-Driven Object Detection with Top-Down Visual Attentions**

- arxiv: [https://arxiv.org/abs/1612.06704](https://arxiv.org/abs/1612.06704)

**Beyond Skip Connections: Top-Down Modulation for Object Detection**

- intro: CMU & UC Berkeley & Google Research
- arxiv: [https://arxiv.org/abs/1612.06851](https://arxiv.org/abs/1612.06851)

## YOLOv2

**YOLO9000: Better, Faster, Stronger**

- arxiv: [https://arxiv.org/abs/1612.08242](https://arxiv.org/abs/1612.08242)
- code: [http://pjreddie.com/yolo9000/](http://pjreddie.com/yolo9000/)
- github(Chainer): [https://github.com/leetenki/YOLOv2](https://github.com/leetenki/YOLOv2)
- github(Keras): [https://github.com/allanzelener/YAD2K](https://github.com/allanzelener/YAD2K)

## DSSD

**DSSD : Deconvolutional Single Shot Detector**

- intro: UNC Chapel Hill & Amazon Inc
- arxiv: [https://arxiv.org/abs/1701.06659](https://arxiv.org/abs/1701.06659)

**Wide-Residual-Inception Networks for Real-time Object Detection**

- intro: Inha University
- arxiv: [https://arxiv.org/abs/1702.01243](https://arxiv.org/abs/1702.01243)

**Attentional Network for Visual Object Detection**

- intro: University of Maryland & Mitsubishi Electric Research Laboratories
- arxiv: [https://arxiv.org/abs/1702.01478](https://arxiv.org/abs/1702.01478)

# Detection From Video

**Learning Object Class Detectors from Weakly Annotated Video**

- intro: CVPR 2012
- paper: [https://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_00905.pdf](https://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_00905.pdf)

**Analysing domain shift factors between videos and images for object detection**

- arxiv: [https://arxiv.org/abs/1501.01186](https://arxiv.org/abs/1501.01186)

**Video Object Recognition**

- slides: [http://vision.princeton.edu/courses/COS598/2015sp/slides/VideoRecog/Video%20Object%20Recognition.pptx](http://vision.princeton.edu/courses/COS598/2015sp/slides/VideoRecog/Video%20Object%20Recognition.pptx)

**Deep Learning for Saliency Prediction in Natural Video**

- intro: Submitted on 12 Jan 2016
- keywords: Deep learning, saliency map, optical flow, convolution network, contrast features
- paper: [https://hal.archives-ouvertes.fr/hal-01251614/document](https://hal.archives-ouvertes.fr/hal-01251614/document)

## T-CNN

**T-CNN: Tubelets with Convolutional Neural Networks for Object Detection from Videos**

- intro: Winning solution in ILSVRC2015 Object Detection from Video(VID) Task
- arxiv: [http://arxiv.org/abs/1604.02532](http://arxiv.org/abs/1604.02532)
- github: [https://github.com/myfavouritekk/T-CNN](https://github.com/myfavouritekk/T-CNN)

**Object Detection from Video Tubelets with Convolutional Neural Networks**

- intro: CVPR 2016 Spotlight paper
- arxiv: [https://arxiv.org/abs/1604.04053](https://arxiv.org/abs/1604.04053)
- paper: [http://www.ee.cuhk.edu.hk/~wlouyang/Papers/KangVideoDet_CVPR16.pdf](http://www.ee.cuhk.edu.hk/~wlouyang/Papers/KangVideoDet_CVPR16.pdf)
- gihtub: [https://github.com/myfavouritekk/vdetlib](https://github.com/myfavouritekk/vdetlib)

**Object Detection in Videos with Tubelets and Multi-context Cues**

- intro: SenseTime Group
- slides: [http://www.ee.cuhk.edu.hk/~xgwang/CUvideo.pdf](http://www.ee.cuhk.edu.hk/~xgwang/CUvideo.pdf)
- slides: [http://image-net.org/challenges/talks/Object%20Detection%20in%20Videos%20with%20Tubelets%20and%20Multi-context%20Cues%20-%20Final.pdf](http://image-net.org/challenges/talks/Object%20Detection%20in%20Videos%20with%20Tubelets%20and%20Multi-context%20Cues%20-%20Final.pdf)

**Context Matters: Refining Object Detection in Video with Recurrent Neural Networks**

- intro: BMVC 2016
- keywords: pseudo-labeler
- arxiv: [http://arxiv.org/abs/1607.04648](http://arxiv.org/abs/1607.04648)
- paper: [http://vision.cornell.edu/se3/wp-content/uploads/2016/07/video_object_detection_BMVC.pdf](http://vision.cornell.edu/se3/wp-content/uploads/2016/07/video_object_detection_BMVC.pdf)

**CNN Based Object Detection in Large Video Images**

- intro: WangTao @ 爱奇艺
- keywords: object retrieval, object detection, scene classification
- slides: [http://on-demand.gputechconf.com/gtc/2016/presentation/s6362-wang-tao-cnn-based-object-detection-large-video-images.pdf](http://on-demand.gputechconf.com/gtc/2016/presentation/s6362-wang-tao-cnn-based-object-detection-large-video-images.pdf)

## Datasets

**YouTube-Objects dataset v2.2**

- homepage: [http://calvin.inf.ed.ac.uk/datasets/youtube-objects-dataset/](http://calvin.inf.ed.ac.uk/datasets/youtube-objects-dataset/)

**ILSVRC2015: Object detection from video (VID)**

- homepage: [http://vision.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid](http://vision.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid)

# Object Detection in 3D

**Vote3Deep: Fast Object Detection in 3D Point Clouds Using Efficient Convolutional Neural Networks**

- arxiv: [https://arxiv.org/abs/1609.06666](https://arxiv.org/abs/1609.06666)

# Object Detection on RGB-D

**Learning Rich Features from RGB-D Images for Object Detection and Segmentation**

- arxiv: [http://arxiv.org/abs/1407.5736](http://arxiv.org/abs/1407.5736)

**Differential Geometry Boosts Convolutional Neural Networks for Object Detection**

- intro: CVPR 2016
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w23/html/Wang_Differential_Geometry_Boosts_CVPR_2016_paper.html](http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w23/html/Wang_Differential_Geometry_Boosts_CVPR_2016_paper.html)

# Salient Object Detection

This task involves predicting the salient regions of an image given by human eye fixations.

**Best Deep Saliency Detection Models (CVPR 2016 & 2015)**

[http://i.cs.hku.hk/~yzyu/vision.html](http://i.cs.hku.hk/~yzyu/vision.html)

**Large-scale optimization of hierarchical features for saliency prediction in natural images**

- paper: [http://coxlab.org/pdfs/cvpr2014_vig_saliency.pdf](http://coxlab.org/pdfs/cvpr2014_vig_saliency.pdf)

**Predicting Eye Fixations using Convolutional Neural Networks**

- paper: [http://www.escience.cn/system/file?fileId=72648](http://www.escience.cn/system/file?fileId=72648)

**Saliency Detection by Multi-Context Deep Learning**

- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhao_Saliency_Detection_by_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhao_Saliency_Detection_by_2015_CVPR_paper.pdf)

**DeepSaliency: Multi-Task Deep Neural Network Model for Salient Object Detection**

- arxiv: [http://arxiv.org/abs/1510.05484](http://arxiv.org/abs/1510.05484)

**SuperCNN: A Superpixelwise Convolutional Neural Network for Salient Object Detection**

- paper: [www.shengfenghe.com/supercnn-a-superpixelwise-convolutional-neural-network-for-salient-object-detection.html](www.shengfenghe.com/supercnn-a-superpixelwise-convolutional-neural-network-for-salient-object-detection.html)

**Shallow and Deep Convolutional Networks for Saliency Prediction**

- arxiv: [http://arxiv.org/abs/1603.00845](http://arxiv.org/abs/1603.00845)
- github: [https://github.com/imatge-upc/saliency-2016-cvpr](https://github.com/imatge-upc/saliency-2016-cvpr)

**Recurrent Attentional Networks for Saliency Detection**

- intro: CVPR 2016. recurrent attentional convolutional-deconvolution network (RACDNN)
- arxiv: [http://arxiv.org/abs/1604.03227](http://arxiv.org/abs/1604.03227)

**Two-Stream Convolutional Networks for Dynamic Saliency Prediction**

- arxiv: [http://arxiv.org/abs/1607.04730](http://arxiv.org/abs/1607.04730)

**Unconstrained Salient Object Detection**

**Unconstrained Salient Object Detection via Proposal Subset Optimization**

![](http://cs-people.bu.edu/jmzhang/images/pasted%20image%201465x373.jpg)

- intro: CVPR 2016
- project page: [http://cs-people.bu.edu/jmzhang/sod.html](http://cs-people.bu.edu/jmzhang/sod.html)
- paper: [http://cs-people.bu.edu/jmzhang/SOD/CVPR16SOD_camera_ready.pdf](http://cs-people.bu.edu/jmzhang/SOD/CVPR16SOD_camera_ready.pdf)
- github: [https://github.com/jimmie33/SOD](https://github.com/jimmie33/SOD)
- caffe model zoo: [https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-object-proposal-models-for-salient-object-detection](https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-object-proposal-models-for-salient-object-detection)

**DHSNet: Deep Hierarchical Saliency Network for Salient Object Detection**

- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf)

**Salient Object Subitizing**

![](http://cs-people.bu.edu/jmzhang/images/frontpage.png?crc=123070793)

- intro: CVPR 2015
- intro: predicting the existence and the number of salient objects in an image using holistic cues
- project page: [http://cs-people.bu.edu/jmzhang/sos.html](http://cs-people.bu.edu/jmzhang/sos.html)
- arxiv: [http://arxiv.org/abs/1607.07525](http://arxiv.org/abs/1607.07525)
- paper: [http://cs-people.bu.edu/jmzhang/SOS/SOS_preprint.pdf](http://cs-people.bu.edu/jmzhang/SOS/SOS_preprint.pdf)
- caffe model zoo: [https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-models-for-salient-object-subitizing](https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-models-for-salient-object-subitizing)

**Deeply-Supervised Recurrent Convolutional Neural Network for Saliency Detection**

- intro: ACMMM 2016. deeply-supervised recurrent convolutional neural network (DSRCNN)
- arxiv: [http://arxiv.org/abs/1608.05177](http://arxiv.org/abs/1608.05177)

**Saliency Detection via Combining Region-Level and Pixel-Level Predictions with CNNs**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1608.05186](http://arxiv.org/abs/1608.05186)

**Edge Preserving and Multi-Scale Contextual Neural Network for Salient Object Detection**

- arxiv: [http://arxiv.org/abs/1608.08029](http://arxiv.org/abs/1608.08029)

**A Deep Multi-Level Network for Saliency Prediction**

- arxiv: [http://arxiv.org/abs/1609.01064](http://arxiv.org/abs/1609.01064)

**Visual Saliency Detection Based on Multiscale Deep CNN Features**

- intro: IEEE Transactions on Image Processing
- arxiv: [http://arxiv.org/abs/1609.02077](http://arxiv.org/abs/1609.02077)

**A Deep Spatial Contextual Long-term Recurrent Convolutional Network for Saliency Detection**

- intro: DSCLRCN
- arxiv: [https://arxiv.org/abs/1610.01708](https://arxiv.org/abs/1610.01708)

**Deeply supervised salient object detection with short connections**

- arxiv: [https://arxiv.org/abs/1611.04849](https://arxiv.org/abs/1611.04849)

**Weakly Supervised Top-down Salient Object Detection**

- intro: Nanyang Technological University
- arxiv: [https://arxiv.org/abs/1611.05345](https://arxiv.org/abs/1611.05345)

**SalGAN: Visual Saliency Prediction with Generative Adversarial Networks**

- project page: [https://imatge-upc.github.io/saliency-salgan-2017/](https://imatge-upc.github.io/saliency-salgan-2017/)
- arxiv: [https://arxiv.org/abs/1701.01081](https://arxiv.org/abs/1701.01081)

**Visual Saliency Prediction Using a Mixture of Deep Neural Networks**

- arxiv: [https://arxiv.org/abs/1702.00372](https://arxiv.org/abs/1702.00372)

**A Fast and Compact Salient Score Regression Network Based on Fully Convolutional Network**

- arxiv: [https://arxiv.org/abs/1702.00615](https://arxiv.org/abs/1702.00615)

## Saliency Detection in Video

**Deep Learning For Video Saliency Detection**

- arxiv: [https://arxiv.org/abs/1702.00871](https://arxiv.org/abs/1702.00871)

## Datasets

**MSRA10K Salient Object Database**

[http://mmcheng.net/msra10k/](http://mmcheng.net/msra10k/)

# Specific Object Deteciton

## Face Deteciton

**Multi-view Face Detection Using Deep Convolutional Neural Networks**

- intro: Yahoo
- arxiv: [http://arxiv.org/abs/1502.02766](http://arxiv.org/abs/1502.02766)

**From Facial Parts Responses to Face Detection: A Deep Learning Approach**

![](http://personal.ie.cuhk.edu.hk/~ys014/projects/Faceness/support/index.png)

- project page: [http://personal.ie.cuhk.edu.hk/~ys014/projects/Faceness/Faceness.html](http://personal.ie.cuhk.edu.hk/~ys014/projects/Faceness/Faceness.html)

**Compact Convolutional Neural Network Cascade for Face Detection**

- arxiv: [http://arxiv.org/abs/1508.01292](http://arxiv.org/abs/1508.01292)
- github: [https://github.com/Bkmz21/FD-Evaluation](https://github.com/Bkmz21/FD-Evaluation)

**Face Detection with End-to-End Integration of a ConvNet and a 3D Model**

- intro: ECCV 2016
- arxiv: [https://arxiv.org/abs/1606.00850](https://arxiv.org/abs/1606.00850)
- github(MXNet): [https://github.com/tfwu/FaceDetection-ConvNet-3D](https://github.com/tfwu/FaceDetection-ConvNet-3D)

**CMS-RCNN: Contextual Multi-Scale Region-based CNN for Unconstrained Face Detection**

- intro: CMU
- arxiv: [https://arxiv.org/abs/1606.05413](https://arxiv.org/abs/1606.05413)

**Finding Tiny Faces**

- intro: CMU
- arxiv: [https://arxiv.org/abs/1612.04402](https://arxiv.org/abs/1612.04402)

**Towards a Deep Learning Framework for Unconstrained Face Detection**

- intro: overlap with CMS-RCNN
- arxiv: [https://arxiv.org/abs/1612.05322](https://arxiv.org/abs/1612.05322)

**Supervised Transformer Network for Efficient Face Detection**

- arxiv: [http://arxiv.org/abs/1607.05477](http://arxiv.org/abs/1607.05477)

### UnitBox

**UnitBox: An Advanced Object Detection Network**

- intro: ACM MM 2016
- arxiv: [http://arxiv.org/abs/1608.01471](http://arxiv.org/abs/1608.01471)

**Bootstrapping Face Detection with Hard Negative Examples**

- author: 万韶华 @ 小米.
- intro: Faster R-CNN, hard negative mining. state-of-the-art on the FDDB dataset
- arxiv: [http://arxiv.org/abs/1608.02236](http://arxiv.org/abs/1608.02236)

**Grid Loss: Detecting Occluded Faces**

- intro: ECCV 2016
- arxiv: [https://arxiv.org/abs/1609.00129](https://arxiv.org/abs/1609.00129)
- paper: [http://lrs.icg.tugraz.at/pubs/opitz_eccv_16.pdf](http://lrs.icg.tugraz.at/pubs/opitz_eccv_16.pdf)
- poster: [http://www.eccv2016.org/files/posters/P-2A-34.pdf](http://www.eccv2016.org/files/posters/P-2A-34.pdf)

**A Multi-Scale Cascade Fully Convolutional Network Face Detector**

- intro: ICPR 2016
- arxiv: [http://arxiv.org/abs/1609.03536](http://arxiv.org/abs/1609.03536)

### MTCNN

**Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks**

**Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks**

![](https://kpzhang93.github.io/MTCNN_face_detection_alignment/support/index.png)

- project page: [https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
- arxiv: [https://arxiv.org/abs/1604.02878](https://arxiv.org/abs/1604.02878)
- github(Matlab): [https://github.com/kpzhang93/MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
- github(MXNet): [https://github.com/pangyupo/mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
- github: [https://github.com/DaFuCoding/MTCNN_Caffe](https://github.com/DaFuCoding/MTCNN_Caffe)
- github(MXNet): [https://github.com/Seanlinx/mtcnn](https://github.com/Seanlinx/mtcnn)

**Face Detection using Deep Learning: An Improved Faster RCNN Approach**

- intro: DeepIR Inc
- arxiv: [https://arxiv.org/abs/1701.08289](https://arxiv.org/abs/1701.08289)

**Faceness-Net: Face Detection through Deep Facial Part Responses**

- intro: An extended version of ICCV 2015 paper
- arxiv: [https://arxiv.org/abs/1701.08393](https://arxiv.org/abs/1701.08393)

### Datasets / Benchmarks

**FDDB: Face Detection Data Set and Benchmark**

- homepage: [http://vis-www.cs.umass.edu/fddb/index.html](http://vis-www.cs.umass.edu/fddb/index.html)
- results: [http://vis-www.cs.umass.edu/fddb/results.html](http://vis-www.cs.umass.edu/fddb/results.html)

**WIDER FACE: A Face Detection Benchmark**

![](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/intro.jpg)

- homepage: [http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
- arxiv: [http://arxiv.org/abs/1511.06523](http://arxiv.org/abs/1511.06523)

## Facial Point / Landmark Detection

**Deep Convolutional Network Cascade for Facial Point Detection**

![](http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/Picture1.png)

- homepage: [http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)
- paper: [http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTcvpr13.pdf](http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTcvpr13.pdf)
- github: [https://github.com/luoyetx/deep-landmark](https://github.com/luoyetx/deep-landmark)

**Facial Landmark Detection by Deep Multi-task Learning**

- intro: ECCV 2014
- project page: [http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html)
- paper: [http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepfacealign.pdf](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepfacealign.pdf)
- github(Matlab): [https://github.com/zhzhanp/TCDCN-face-alignment](https://github.com/zhzhanp/TCDCN-face-alignment)

**A Recurrent Encoder-Decoder Network for Sequential Face Alignment**

- intro: ECCV 2016
- arxiv: [https://arxiv.org/abs/1608.05477](https://arxiv.org/abs/1608.05477)

**Detecting facial landmarks in the video based on a hybrid framework**

- arxiv: [http://arxiv.org/abs/1609.06441](http://arxiv.org/abs/1609.06441)

**Deep Constrained Local Models for Facial Landmark Detection**

- arxiv: [https://arxiv.org/abs/1611.08657](https://arxiv.org/abs/1611.08657)

**Effective face landmark localization via single deep network**

- arxiv: [https://arxiv.org/abs/1702.02719](https://arxiv.org/abs/1702.02719)

## People Detection

**End-to-end people detection in crowded scenes**

![](/assets/object-detection-materials/end_to_end_people_detection_in_crowded_scenes.jpg)

- arxiv: [http://arxiv.org/abs/1506.04878](http://arxiv.org/abs/1506.04878)
- github: [https://github.com/Russell91/reinspect](https://github.com/Russell91/reinspect)
- ipn: [http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb](http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb)

**Detecting People in Artwork with CNNs**

- intro: ECCV 2016 Workshops
- arxiv: [https://arxiv.org/abs/1610.08871](https://arxiv.org/abs/1610.08871)

**Deep Multi-camera People Detection**

- arxiv: [https://arxiv.org/abs/1702.04593](https://arxiv.org/abs/1702.04593)

## Person Head Detection

**Context-aware CNNs for person head detection**

- arxiv: [http://arxiv.org/abs/1511.07917](http://arxiv.org/abs/1511.07917)
- github: [https://github.com/aosokin/cnn_head_detection](https://github.com/aosokin/cnn_head_detection)

## Pedestrian Detection

**Pedestrian Detection aided by Deep Learning Semantic Tasks**

- intro: CVPR 2015
- project page: [http://mmlab.ie.cuhk.edu.hk/projects/TA-CNN/](http://mmlab.ie.cuhk.edu.hk/projects/TA-CNN/)
- paper: [http://arxiv.org/abs/1412.0069](http://arxiv.org/abs/1412.0069)

**Deep Learning Strong Parts for Pedestrian Detection**

- intro: ICCV 2015. CUHK. DeepParts
- intro: Achieving 11.89% average miss rate on Caltech Pedestrian Dataset
- paper: [http://personal.ie.cuhk.edu.hk/~pluo/pdf/tianLWTiccv15.pdf](http://personal.ie.cuhk.edu.hk/~pluo/pdf/tianLWTiccv15.pdf)

**Deep convolutional neural networks for pedestrian detection**

- arxiv: [http://arxiv.org/abs/1510.03608](http://arxiv.org/abs/1510.03608)
- github: [https://github.com/DenisTome/DeepPed](https://github.com/DenisTome/DeepPed)

**Scale-aware Fast R-CNN for Pedestrian Detection**

- arxiv: [https://arxiv.org/abs/1510.08160](https://arxiv.org/abs/1510.08160)

**New algorithm improves speed and accuracy of pedestrian detection**

- blog: [http://www.eurekalert.org/pub_releases/2016-02/uoc--nai020516.php](http://www.eurekalert.org/pub_releases/2016-02/uoc--nai020516.php)

**Pushing the Limits of Deep CNNs for Pedestrian Detection**

- intro: "set a new record on the Caltech pedestrian dataset, lowering the log-average miss rate from 11.7% to 8.9%"
- arxiv: [http://arxiv.org/abs/1603.04525](http://arxiv.org/abs/1603.04525)

**A Real-Time Deep Learning Pedestrian Detector for Robot Navigation**

- arxiv: [http://arxiv.org/abs/1607.04436](http://arxiv.org/abs/1607.04436)

**A Real-Time Pedestrian Detector using Deep Learning for Human-Aware Navigation**

- arxiv: [http://arxiv.org/abs/1607.04441](http://arxiv.org/abs/1607.04441)

**Is Faster R-CNN Doing Well for Pedestrian Detection?**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1607.07032](http://arxiv.org/abs/1607.07032)
- github: [https://github.com/zhangliliang/RPN_BF/tree/RPN-pedestrian](https://github.com/zhangliliang/RPN_BF/tree/RPN-pedestrian)

**Reduced Memory Region Based Deep Convolutional Neural Network Detection**

- intro: IEEE 2016 ICCE-Berlin
- arxiv: [http://arxiv.org/abs/1609.02500](http://arxiv.org/abs/1609.02500)

**Fused DNN: A deep neural network fusion approach to fast and robust pedestrian detection**

- arxiv: [https://arxiv.org/abs/1610.03466](https://arxiv.org/abs/1610.03466)

**Multispectral Deep Neural Networks for Pedestrian Detection**

- intro: BMVC 2016 oral
- arxiv: [https://arxiv.org/abs/1611.02644](https://arxiv.org/abs/1611.02644)

## Vehicle Detection

**DAVE: A Unified Framework for Fast Vehicle Detection and Annotation**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1607.04564](http://arxiv.org/abs/1607.04564)

**Evolving Boxes for fast Vehicle Detection**

- arxiv: [https://arxiv.org/abs/1702.00254](https://arxiv.org/abs/1702.00254)

## Traffic-Sign Detection

**Traffic-Sign Detection and Classification in the Wild**

- project page(code+dataset): [http://cg.cs.tsinghua.edu.cn/traffic-sign/](http://cg.cs.tsinghua.edu.cn/traffic-sign/)
- paper: [http://120.52.73.11/www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf](http://120.52.73.11/www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf)
- code & model: [http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/newdata0411.zip](http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/newdata0411.zip)

## Boundary / Edge / Contour Detection

**Holistically-Nested Edge Detection**

![](https://camo.githubusercontent.com/da32e7e3275c2a9693dd2a6925b03a1151e2b098/687474703a2f2f70616765732e756373642e6564752f7e7a74752f6865642e6a7067)

- intro: ICCV 2015, Marr Prize
- paper: [http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.pdf](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.pdf)
- arxiv: [http://arxiv.org/abs/1504.06375](http://arxiv.org/abs/1504.06375)
- github: [https://github.com/s9xie/hed](https://github.com/s9xie/hed)

**Unsupervised Learning of Edges**

- intro: CVPR 2016. Facebook AI Research
- arxiv: [http://arxiv.org/abs/1511.04166](http://arxiv.org/abs/1511.04166)
- zn-blog: [http://www.leiphone.com/news/201607/b1trsg9j6GSMnjOP.html](http://www.leiphone.com/news/201607/b1trsg9j6GSMnjOP.html)

**Pushing the Boundaries of Boundary Detection using Deep Learning**

- arxiv: [http://arxiv.org/abs/1511.07386](http://arxiv.org/abs/1511.07386)

**Convolutional Oriented Boundaries**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1608.02755](http://arxiv.org/abs/1608.02755)

**Convolutional Oriented Boundaries: From Image Segmentation to High-Level Tasks**

- project page: [http://www.vision.ee.ethz.ch/~cvlsegmentation/](http://www.vision.ee.ethz.ch/~cvlsegmentation/)
- arxiv: [https://arxiv.org/abs/1701.04658](https://arxiv.org/abs/1701.04658)

**Richer Convolutional Features for Edge Detection**

- intro: richer convolutional features (RCF)
- arxiv: [https://arxiv.org/abs/1612.02103](https://arxiv.org/abs/1612.02103)

## Skeleton Detection

**Object Skeleton Extraction in Natural Images by Fusing Scale-associated Deep Side Outputs**

![](https://camo.githubusercontent.com/88a65f132aa4ae4b0477e3ad02c13cdc498377d9/687474703a2f2f37786e37777a2e636f6d312e7a302e676c622e636c6f7564646e2e636f6d2f44656570536b656c65746f6e2e706e673f696d61676556696577322f322f772f353030)

- arxiv: [http://arxiv.org/abs/1603.09446](http://arxiv.org/abs/1603.09446)
- github: [https://github.com/zeakey/DeepSkeleton](https://github.com/zeakey/DeepSkeleton)

**DeepSkeleton: Learning Multi-task Scale-associated Deep Side Outputs for Object Skeleton Extraction in Natural Images**

- arxiv: [http://arxiv.org/abs/1609.03659](http://arxiv.org/abs/1609.03659)

## Fruit Detection

**Deep Fruit Detection in Orchards**

- arxiv: [https://arxiv.org/abs/1610.03677](https://arxiv.org/abs/1610.03677)

**Image Segmentation for Fruit Detection and Yield Estimation in Apple Orchards**

- intro: The Journal of Field Robotics in May 2016
- project page: [http://confluence.acfr.usyd.edu.au/display/AGPub/](http://confluence.acfr.usyd.edu.au/display/AGPub/)
- arxiv: [https://arxiv.org/abs/1610.08120](https://arxiv.org/abs/1610.08120)

## Others

**Deep Deformation Network for Object Landmark Localization**

- arxiv: [http://arxiv.org/abs/1605.01014](http://arxiv.org/abs/1605.01014)

**Fashion Landmark Detection in the Wild**

- arxiv: [http://arxiv.org/abs/1608.03049](http://arxiv.org/abs/1608.03049)

**Deep Learning for Fast and Accurate Fashion Item Detection**

- intro: Kuznech Inc.
- intro: MultiBox and Fast R-CNN
- paper: [https://kddfashion2016.mybluemix.net/kddfashion_finalSubmissions/Deep%20Learning%20for%20Fast%20and%20Accurate%20Fashion%20Item%20Detection.pdf](https://kddfashion2016.mybluemix.net/kddfashion_finalSubmissions/Deep%20Learning%20for%20Fast%20and%20Accurate%20Fashion%20Item%20Detection.pdf)

**Visual Relationship Detection with Language Priors**

- intro: ECCV 2016 oral
- paper: [https://cs.stanford.edu/people/ranjaykrishna/vrd/vrd.pdf](https://cs.stanford.edu/people/ranjaykrishna/vrd/vrd.pdf)
- github: [https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection](https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection)

**OSMDeepOD - OSM and Deep Learning based Object Detection from Aerial Imagery (formerly known as "OSM-Crosswalk-Detection")**

![](https://raw.githubusercontent.com/geometalab/OSMDeepOD/master/imgs/process.png)

- github: [https://github.com/geometalab/OSMDeepOD](https://github.com/geometalab/OSMDeepOD)

**Selfie Detection by Synergy-Constraint Based Convolutional Neural Network**

- intro:  IEEE SITIS 2016
- arxiv: [https://arxiv.org/abs/1611.04357](https://arxiv.org/abs/1611.04357)

**Associative Embedding:End-to-End Learning for Joint Detection and Grouping**

- arxiv: [https://arxiv.org/abs/1611.05424](https://arxiv.org/abs/1611.05424)

**Deep Cuboid Detection: Beyond 2D Bounding Boxes**

- intro: CMU & Magic Leap
- arxiv: [https://arxiv.org/abs/1611.10010](https://arxiv.org/abs/1611.10010)

**Automatic Model Based Dataset Generation for Fast and Accurate Crop and Weeds Detection**

- arxiv: [https://arxiv.org/abs/1612.03019](https://arxiv.org/abs/1612.03019)

**Deep Learning Logo Detection with Data Expansion by Synthesising Context**

- arxiv: [https://arxiv.org/abs/1612.09322](https://arxiv.org/abs/1612.09322)

**Pixel-wise Ear Detection with Convolutional Encoder-Decoder Networks**

- arxiv: [https://arxiv.org/abs/1702.00307](https://arxiv.org/abs/1702.00307)

**Automatic Handgun Detection Alarm in Videos Using Deep Learning**

- arxiv: [https://arxiv.org/abs/1702.05147](https://arxiv.org/abs/1702.05147)
- results: [https://github.com/SihamTabik/Pistol-Detection-in-Videos](https://github.com/SihamTabik/Pistol-Detection-in-Videos)

# Object Proposal

**DeepProposal: Hunting Objects by Cascading Deep Convolutional Layers**

- arxiv: [http://arxiv.org/abs/1510.04445](http://arxiv.org/abs/1510.04445)
- github: [https://github.com/aghodrati/deepproposal](https://github.com/aghodrati/deepproposal)

**Scale-aware Pixel-wise Object Proposal Networks**

- intro: IEEE Transactions on Image Processing
- arxiv: [http://arxiv.org/abs/1601.04798](http://arxiv.org/abs/1601.04798)

**Attend Refine Repeat: Active Box Proposal Generation via In-Out Localization**

- intro: BMVC 2016. AttractioNet
- arxiv: [https://arxiv.org/abs/1606.04446](https://arxiv.org/abs/1606.04446)
- github: [https://github.com/gidariss/AttractioNet](https://github.com/gidariss/AttractioNet)

**Learning to Segment Object Proposals via Recursive Neural Networks**

- arxiv: [https://arxiv.org/abs/1612.01057](https://arxiv.org/abs/1612.01057)

# Localization

**Beyond Bounding Boxes: Precise Localization of Objects in Images**

- intro: PhD Thesis
- homepage: [http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.html](http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.html)
- phd-thesis: [http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.pdf](http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.pdf)
- github("SDS using hypercolumns"): [https://github.com/bharath272/sds](https://github.com/bharath272/sds)

**Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning**

- arxiv: [http://arxiv.org/abs/1503.00949](http://arxiv.org/abs/1503.00949)

**Weakly Supervised Object Localization Using Size Estimates**

- arxiv: [http://arxiv.org/abs/1608.04314](http://arxiv.org/abs/1608.04314)

**Active Object Localization with Deep Reinforcement Learning**

- intro: ICCV 2015
- keywords: Markov Decision Process
- arxiv: [https://arxiv.org/abs/1511.06015](https://arxiv.org/abs/1511.06015)

**Localizing objects using referring expressions**

- intro: ECCV 2016
- keywords: LSTM, multiple instance learning (MIL)
- paper: [http://www.umiacs.umd.edu/~varun/files/refexp-ECCV16.pdf](http://www.umiacs.umd.edu/~varun/files/refexp-ECCV16.pdf)
- github: [https://github.com/varun-nagaraja/referring-expressions](https://github.com/varun-nagaraja/referring-expressions)

**LocNet: Improving Localization Accuracy for Object Detection**

- arxiv: [http://arxiv.org/abs/1511.07763](http://arxiv.org/abs/1511.07763)
- github: [https://github.com/gidariss/LocNet](https://github.com/gidariss/LocNet)

**Learning Deep Features for Discriminative Localization**

![](http://cnnlocalization.csail.mit.edu/framework.jpg)

- homepage: [http://cnnlocalization.csail.mit.edu/](http://cnnlocalization.csail.mit.edu/)
- arxiv: [http://arxiv.org/abs/1512.04150](http://arxiv.org/abs/1512.04150)
- github(Tensorflow): [https://github.com/jazzsaxmafia/Weakly_detector](https://github.com/jazzsaxmafia/Weakly_detector)
- github: [https://github.com/metalbubble/CAM](https://github.com/metalbubble/CAM)
- github: [https://github.com/tdeboissiere/VGG16CAM-keras](https://github.com/tdeboissiere/VGG16CAM-keras)

**ContextLocNet: Context-Aware Deep Network Models for Weakly Supervised Localization**

![](http://www.di.ens.fr/willow/research/contextlocnet/model.png)

- intro: ECCV 2016
- project page: [http://www.di.ens.fr/willow/research/contextlocnet/](http://www.di.ens.fr/willow/research/contextlocnet/)
- arxiv: [http://arxiv.org/abs/1609.04331](http://arxiv.org/abs/1609.04331)
- github: [https://github.com/vadimkantorov/contextlocnet](https://github.com/vadimkantorov/contextlocnet)

# Tutorials / Talks

**Convolutional Feature Maps: Elements of efficient (and accurate) CNN-based object detection**

- slides: [http://research.microsoft.com/en-us/um/people/kahe/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf](http://research.microsoft.com/en-us/um/people/kahe/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf)

**Towards Good Practices for Recognition & Detection**

- intro: Hikvision Research Institute. Supervised Data Augmentation (SDA)
- slides: [http://image-net.org/challenges/talks/2016/Hikvision_at_ImageNet_2016.pdf](http://image-net.org/challenges/talks/2016/Hikvision_at_ImageNet_2016.pdf)

# Projects

**TensorBox: a simple framework for training neural networks to detect objects in images**

- intro: "The basic model implements the simple and robust GoogLeNet-OverFeat algorithm. 
We additionally provide an implementation of the [ReInspect](https://github.com/Russell91/ReInspect/) algorithm"
- github: [https://github.com/Russell91/TensorBox](https://github.com/Russell91/TensorBox)

**Object detection in torch: Implementation of some object detection frameworks in torch**

- github: [https://github.com/fmassa/object-detection.torch](https://github.com/fmassa/object-detection.torch)

**Using DIGITS to train an Object Detection network**

- github: [https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/README.md](https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/README.md)

**FCN-MultiBox Detector**

- intro: Full convolution MultiBox Detector (like SSD) implemented in Torch.
- github: [https://github.com/teaonly/FMD.torch](https://github.com/teaonly/FMD.torch)

**KittiBox: A car detection model implemented in Tensorflow.**

- keywords: MultiNet
- intro: KittiBox is a collection of scripts to train out model FastBox on the Kitti Object Detection Dataset
- github: [https://github.com/MarvinTeichmann/KittiBox](https://github.com/MarvinTeichmann/KittiBox)

# Blogs

**Convolutional Neural Networks for Object Detection**

[http://rnd.azoft.com/convolutional-neural-networks-object-detection/](http://rnd.azoft.com/convolutional-neural-networks-object-detection/)

**Introducing automatic object detection to visual search (Pinterest)**

- keywords: Faster R-CNN
- blog: [https://engineering.pinterest.com/blog/introducing-automatic-object-detection-visual-search](https://engineering.pinterest.com/blog/introducing-automatic-object-detection-visual-search)
- demo: [https://engineering.pinterest.com/sites/engineering/files/Visual%20Search%20V1%20-%20Video.mp4](https://engineering.pinterest.com/sites/engineering/files/Visual%20Search%20V1%20-%20Video.mp4)
- review: [https://news.developer.nvidia.com/pinterest-introduces-the-future-of-visual-search/?mkt_tok=eyJpIjoiTnpaa01UWXpPRE0xTURFMiIsInQiOiJJRjcybjkwTmtmallORUhLOFFFODBDclFqUlB3SWlRVXJXb1MrQ013TDRIMGxLQWlBczFIeWg0TFRUdnN2UHY2ZWFiXC9QQVwvQzBHM3B0UzBZblpOSmUyU1FcLzNPWXI4cml2VERwTTJsOFwvOEk9In0%3D](https://news.developer.nvidia.com/pinterest-introduces-the-future-of-visual-search/?mkt_tok=eyJpIjoiTnpaa01UWXpPRE0xTURFMiIsInQiOiJJRjcybjkwTmtmallORUhLOFFFODBDclFqUlB3SWlRVXJXb1MrQ013TDRIMGxLQWlBczFIeWg0TFRUdnN2UHY2ZWFiXC9QQVwvQzBHM3B0UzBZblpOSmUyU1FcLzNPWXI4cml2VERwTTJsOFwvOEk9In0%3D)

**Deep Learning for Object Detection with DIGITS**

- blog: [https://devblogs.nvidia.com/parallelforall/deep-learning-object-detection-digits/](https://devblogs.nvidia.com/parallelforall/deep-learning-object-detection-digits/)

**Analyzing The Papers Behind Facebook's Computer Vision Approach**

- keywords: DeepMask, SharpMask, MultiPathNet
- blog: [https://adeshpande3.github.io/adeshpande3.github.io/Analyzing-the-Papers-Behind-Facebook's-Computer-Vision-Approach/](https://adeshpande3.github.io/adeshpande3.github.io/Analyzing-the-Papers-Behind-Facebook's-Computer-Vision-Approach/)

**Easily Create High Quality Object Detectors with Deep Learning**

- intro: dlib v19.2
- blog: [http://blog.dlib.net/2016/10/easily-create-high-quality-object.html](http://blog.dlib.net/2016/10/easily-create-high-quality-object.html)

**How to Train a Deep-Learned Object Detection Model in the Microsoft Cognitive Toolkit**

- blog: [https://blogs.technet.microsoft.com/machinelearning/2016/10/25/how-to-train-a-deep-learned-object-detection-model-in-cntk/](https://blogs.technet.microsoft.com/machinelearning/2016/10/25/how-to-train-a-deep-learned-object-detection-model-in-cntk/)
- github: [https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection/FastRCNN](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection/FastRCNN)

**Object Detection in Satellite Imagery, a Low Overhead Approach**

- part 1: [https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7#.2csh4iwx9](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7#.2csh4iwx9)
- part 2: [https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-ii-893f40122f92#.f9b7dgf64](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-ii-893f40122f92#.f9b7dgf64)

**You Only Look Twice — Multi-Scale Object Detection in Satellite Imagery With Convolutional Neural Networks**

- part 1: [https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-38dad1cf7571#.fmmi2o3of](https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-38dad1cf7571#.fmmi2o3of)
- part 2: [https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-34f72f659588#.nwzarsz1t](https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-34f72f659588#.nwzarsz1t)

**Faster R-CNN Pedestrian and Car Detection**

- blog: [https://bigsnarf.wordpress.com/2016/11/07/faster-r-cnn-pedestrian-and-car-detection/](https://bigsnarf.wordpress.com/2016/11/07/faster-r-cnn-pedestrian-and-car-detection/)
- ipn: [https://gist.github.com/bigsnarfdude/2f7b2144065f6056892a98495644d3e0#file-demo_faster_rcnn_notebook-ipynb](https://gist.github.com/bigsnarfdude/2f7b2144065f6056892a98495644d3e0#file-demo_faster_rcnn_notebook-ipynb)
- github: [https://github.com/bigsnarfdude/Faster-RCNN_TF](https://github.com/bigsnarfdude/Faster-RCNN_TF)

**Small U-Net for vehicle detection**

- blog: [https://medium.com/@vivek.yadav/small-u-net-for-vehicle-detection-9eec216f9fd6#.md4u80kad](https://medium.com/@vivek.yadav/small-u-net-for-vehicle-detection-9eec216f9fd6#.md4u80kad)



--------------------
*文章待整理* <br>
*许多资料搬运自[handong1587](https://handong1587.github.io/)*