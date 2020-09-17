# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, max_iou_distance=0.7, max_age=70, n_init=3):
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.tracks = []
        self._next_id = 1

    def predict(self):
        """
        This function should be called once every time step,before 'update'.
        """
        for track in self.tracks:
            track.predict()

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade. 级联匹配
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        #print(matches,unmatched_tracks,unmatched_detections)
        # Update track set.
        # 针对match的,要使用检测结果去更新相应的tracker参数
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        #利用每个检测框来创建其对应的新tracker,毕竟这些个检测框没有匹配上嘛
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        # 更新tracks,踢掉删除了的
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _match(self, detections):
        matches, unmatched_tracks, unmatched_detections = \
            linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance, 
                self.tracks, detections)        
        return matches, unmatched_tracks, unmatched_detections

    #利用检测框创建一条新的track
    def _initiate_track(self, detection):
        #根据初始检测位置初始化新kf滤波器的mean和variance
        mean = detection.to_xyah() #a:aspect,h:height 
        self.tracks.append(Track(mean, self._next_id, self.n_init, self.max_age))
        self._next_id += 1 #这个就是追踪的id计数
