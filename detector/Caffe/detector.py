# -*- coding: utf-8 -*-
# vim: set tabstop=4
# vim: expandtab
# vim: set shiftwidth=4
import cv2
import caffe 
import numpy as np 

class SSD(object):
    def __init__(self, prototxt, weight, score_thresh):
        self._score_thresh = score_thresh 
        self._net = caffe.Net(prototxt, weights=weight, phase=caffe.TEST)
        _,self._C,self._H,self._W = self._net.blobs['data'].shape 
        self._layer_name_input = list(self._net.blobs.keys())[0]
        self._layer_name_output = list(self._net.blobs.keys())[-1]

    def __call__(self, img):
        _img_h, _img_w = img.shape[:2] #org image size 
        cls_ids = []
        cls_conf = []
        bbox_xywh = []

        img = caffe.io.resize(img, (self._H,self._W,self._C))
        data = img.transpose(2,0,1)
        self._net.blobs[self._layer_name_input].data[...] = data
        out = self._net.forward()
        for box in out[self._layer_name_output][0][0]:
            score = round(box[2],2)
            if score > self._score_thresh and box[1] != 0: # skip background
                x0 = max(box[3] * _img_w, 0)
                y0 = max(box[4] * _img_h, 0)
                x1 = min(box[5] * _img_w, _img_w)
                y1 = min(box[6] * _img_h, _img_h)
                cx = (x0+x1)/2.
                cy = (y0+y1)/2.
                #just for test
                # x0 = cx - 160
                # y0 = cy - 160
                # x1 = cx + 160
                # y1 = cy + 160
                # x0 = max(x0, 0)
                # y0 = max(y0, 0)
                # x1 = min(x1, _img_w)
                # y1 = min(y1, _img_h)

                bbox_xywh.append([cx,cy,x1-x0,y1-y0])
                cls_conf.append(score)
                cls_ids.append(box[1])
                
        return np.array(bbox_xywh),np.array(cls_conf),np.array(cls_ids)


if __name__ == "__main__":
    path='/home/data4t1/tsd/japan-bg/2020-OlympicsHost-DrivingDowntown-RightDown-1920x640-NoBigTs/2020-OlympicsHost-DrivingDowntown_01059.jpg'
    path='/home/zuosi/deep_sort_pytorch/test_videos/2017_0104_204404_052.MOV'
    path='/home/zuosi/deep_sort_pytorch/test_videos/passenger/20200914/VideoMainStreamChn2_200914122000.h264.mp4'

    proto='./pedestrian/deploy.prototxt'
    weight='./pedestrian/PED_SSD_H5A15-KYSiNetV3_VGGK735_8X_FPN_640x360_anchor15x1_C2_0914_snapshot_iter_672000.caffemodel'

    proto='/home/zuosi/deep_sort_pytorch/detector/Caffe/passenger_counting_0911/deploy.prototxt'
    weight='/home/zuosi/deep_sort_pytorch/detector/Caffe/passenger_counting_0911/FACE_SSD-KYSiNetV4_VGG_8X_FPN_H4A16x1_C2_320x180_0728_snapshot_iter_302000.caffemodel'
    # proto = "/home/liyang/models/pc_0914/model-D2_TLD_SSD-KYSiNetV4_VGG_8X_H3_v2_FPN_H3A16x3_C3_320x180_20200914/deploy.prototxt"
    # weight = "/home/liyang/models/pc_0914/model-D2_TLD_SSD-KYSiNetV4_VGG_8X_H3_v2_FPN_H3A16x3_C3_320x180_20200914/backup/D2_TLD_SSD-KYSiNetV4_VGG_8X_H3_v2_FPN_H3A16x3_C3_320x180_20200914_snapshot_iter_202000.caffemodel"

    ssd_ = SSD(proto,weight,0.8)

    vdo = cv2.VideoCapture()
    vdo.open(path)
    assert vdo.isOpened()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    im_width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('out.avi', fourcc, 20.0, (im_width,im_height))

    def _xywh2xyxy(box):
        box[0] = box[0]-box[2]/2
        box[1] = box[1]-box[3]/2
        box[2] = box[0]+box[2]
        box[3] = box[1]+box[3]
        box = [int(e) for e in box]
        return box 
         
    j = 0
    while vdo.grab(): #necessary
        j += 1
        _, frame = vdo.retrieve()
        print("#fr: {:d}".format(j))
        bbox_xywh,_,_ = ssd_(frame)
        for b in bbox_xywh:
            b = _xywh2xyxy(b)
            cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),(255,0,0),1,0)
        out.write(frame)
        #cv2.imshow('preview', frame)
        #if (cv2.waitKey(-1) & 0xFF) != ord('q'):
        #    continue
        #else:
        #    break
