# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
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

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
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

        # Update distance metric.
        # 更新已经确认的track的特征集
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        """
        把这些active_target之前保存的feature(之前每帧只要能匹配上,都会把与之匹配的det的feature保存
        下来)用于更新卡尔曼滤波的distance metric
        f1,f2,f3,f4,f5,f6,f7,...
        id1,id1,id1,id2,id2,id2,id2,...
        """
        for track in self.tracks:
            if not track.is_confirmed(): 
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            #这个把特征集清空了...更新已经确认的track的特征集
            track.features = [] 
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match_v2(self, detections):
        #计算当前帧每个新检测结果的深度特征与这一层中每个track已保存的特征集之间的余弦距离矩阵
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            #运动信息约束
            # cost_matrix = linear_assignment.gate_cost_matrix(
            #     self.kf, cost_matrix, tracks, dets, track_indices,
            #     detection_indices)

            return cost_matrix
        
        matches_a, unmatched_tracks, unmatched_detections = \
            linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance, 
                self.tracks, detections)        

        # Split track set into confirmed and unconfirmed tracks.
        unmatched_unconfirmed_tracks = [
            i for i in unmatched_tracks if not self.tracks[i].is_confirmed()]
        unmatched_confirmed_tracks = [
            i for i in unmatched_tracks if self.tracks[i].is_confirmed()]
        # Associate confirmed tracks using appearance features.
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, unmatched_confirmed_tracks, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_unconfirmed_tracks + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    #default match policy
    def _match(self, detections):
        #计算当前帧每个新检测结果的深度特征与这一层中每个track已保存的特征集之间的余弦距离矩阵
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            #运动信息约束
            # cost_matrix = linear_assignment.gate_cost_matrix(
            #     self.kf, cost_matrix, tracks, dets, track_indices,
            #     detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1] 
        #等于1的部分送到iou_track_candidates里面去了
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance, 
                self.tracks, detections, iou_track_candidates, unmatched_detections)        

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    #利用检测框创建一条新的track
    def _initiate_track(self, detection):
        #根据初始检测位置初始化新kf滤波器的mean和variance
        mean = detection.to_xyah() #a:aspect,h:height 
        self.tracks.append(Track(mean, self._next_id, self.n_init, 
            self.max_age, detection.feature))
        self._next_id += 1 #这个就是追踪的id计数
