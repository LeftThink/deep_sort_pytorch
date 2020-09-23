import numpy as np
import cv2
from easydict import EasyDict as edict 

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
tracks = edict({})

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


in_passenger_cnt = 0
out_passenger_cnt = 0
passenger_cnt_stat = edict({})

def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    global in_passenger_cnt,out_passenger_cnt
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        cx = (x1+x2)//2
        cy = (y1+y2)//2 

        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        key = str(id)
        if tracks.get(key) is None:
            tracks[key] = []
        else:
            tracks[key].insert(0,(cx,cy))
            if len(tracks[key]) > 15:
                tracks[key].pop()
        color = compute_color_for_labels(id)
        for i in range(len(tracks[key])-1):
            #cv2.circle(img,p,2,color,5) 
            cur_point = tracks[key][i]
            nxt_point = tracks[key][i+1]
            cv2.line(img, cur_point, nxt_point, color, 2)

        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0] #获取文字的宽度和高度
        cv2.rectangle(img,(x1,y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1,y1),(x1+t_size[0]+3,y1+t_size[1]+4),color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    
        h,w,_ = img.shape
        """
        # up_line = int(1/2.*h - 1)
        # down_line = int(1/2.*h + 1)
        # cv2.line(img, (0,up_line),(w,up_line),(0,255,0),3)
        # cv2.line(img, (0,down_line),(w,down_line),(0,255,0),3)

        # #stat in/out passenger 
        # if passenger_cnt_stat.get(key) is None:
        #     if cy < up_line: 
        #         passenger_cnt_stat[key] = {'up':1,'mid':0,'down':0}
        #     elif cy > down_line:
        #         passenger_cnt_stat[key] = {'up':0,'mid':0,'down':1}
        # else:
        #     val = passenger_cnt_stat[key]
        #     if cy < up_line:
        #         if val['down'] and val['mid'] and not val['up']:
        #             in_passenger_cnt += 1
        #             val['up'] = 1
        #             val['mid'] = 0
        #             val['down'] = 0
        #         else:
        #             val['up'] = 1
        #     elif cy > down_line:
        #         if val['up'] and val['mid'] and not val['down']:
        #             out_passenger_cnt += 1
        #             val['down'] = 1
        #             val['mid'] = 0
        #             val['up'] = 0
        #         else:
        #             val['down'] = 1
        #     else:
        #         val['mid'] = 1
        """
        line = int(1/2.*h)
        cv2.line(img, (0,line),(w,line),(0,255,0),3)
        if passenger_cnt_stat.get(key) is None:
            if cy <= line:
                passenger_cnt_stat[key] = {'down':0,'up':1}
            else:
                passenger_cnt_stat[key] = {'down':1,'up':0}
        else:
            val = passenger_cnt_stat[key]
            if cy <= line:
                if val['down'] and not val['up']:
                    in_passenger_cnt += 1
                    val['up'] = 1
                else:
                    val['up'] = 1
            else:
                if val['up'] and not val['down']:
                    out_passenger_cnt += 1
                    val['down'] = 1
                else:
                    val['down'] =1 

        in_cnt = "in:{:d}".format(in_passenger_cnt)
        t_size = cv2.getTextSize(in_cnt, cv2.FONT_HERSHEY_PLAIN, 5 , 5)[0] #获取文字的宽度和高度
        cv2.putText(img,in_cnt,(w-t_size[0]-100,50), cv2.FONT_HERSHEY_PLAIN, 4, [255,255,255], 4)

        out_cnt = "out:{:d}".format(out_passenger_cnt)
        cv2.putText(img,out_cnt,(w-t_size[0]-100,t_size[1]+50+20), cv2.FONT_HERSHEY_PLAIN, 4, [255,255,255], 4)

    return img



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
