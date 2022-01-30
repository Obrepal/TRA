import freenect
import numpy as np
import cv2
import frame_convert2


def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
def get_depth():
    return freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)[0]

def get_dvideo():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)[0])