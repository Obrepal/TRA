#import the necessary modules
import freenect
import cv2
import numpy as np

import frame_convert2



#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

def get_depth():

    return freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)[0]

def get_dvideo():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)[0])


if __name__ == "__main__":


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_rgb = cv2.VideoWriter('tennis_ball_rgb_3.avi', fourcc, 30.0, (640,  480))
    out_depth = cv2.VideoWriter('tennis_ball_depth_3.avi', fourcc, 30.0, (640, 480))
    out_dvideo = cv2.VideoWriter('tennis_ball_pretty_3.avi', fourcc, 30.0, (640, 480))



    cap = cv2.VideoCapture(1)  # video capture source camera (Here webcam of laptop)
    while 1:
        #ret, frame = cap.read()

        #get a frame from RGB camera
        frame = get_video()
        #get a frame from depth sensor
        depth = get_depth()

        dvideo = get_dvideo()
        #display RGB image
        cv2.imshow('RGB image',frame)
       #cv2.findfindChessboardCorners(frame, (2,3))
        #display depth image
        cv2.imshow('Depth image',depth)

        cv2.imshow('Video depth image', dvideo)
    
        # out_rgb.write(frame)
        # #out_depth.write(depth)
        # out_dvideo.write(dvideo)

        # #quit program when 'esc' key is pressed
        # save on pressing # 'y'
        if cv2.waitKey(1) & 0xFF == ord('y'):
            cv2.imwrite('/Users/igorobrepalski/Desktop/PD1_rgb.png', frame)
            cv2.imwrite('/Users/igorobrepalski/Desktop/PD1_depth.png', depth)


        k = cv2.waitKey(5) & 0xFF
        if cv2.waitKey(33) == ord('a'):
            break
        if k == 0:
            break
    cv2.destroyAllWindows()