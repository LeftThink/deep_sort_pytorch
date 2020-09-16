from .YOLOv3 import YOLOv3
from .Caffe import SSD

__all__ = ['build_detector']

def build_detector(cfg, use_cuda):
    """
    return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES, 
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, 
                    is_xywh=True, use_cuda=use_cuda)
    """
    return SSD(cfg.SSD.PROTOTXT,cfg.SSD.WEIGHT,cfg.SSD.SCORE_THRESH) 
