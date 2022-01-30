import cv2
import numpy as np
from collections import deque
import argparse
import matplotlib.pyplot as plt

from kinectImage import get_depth,get_dvideo,get_video
from tennisBall import tennisBall
from cameraMatrix import cameraMatrix
from KalmanFunctions import *
#z pliku
#cap = cv2.VideoCapture('tennis_ball_rgb_3.avi')
#cap = cv2.VideoCapture('tennis_ball_rgb_3.avi')

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])
counter = 0

coord = []
coord_z = []
coord_full = []

# z kamery 
cap = cv2.VideoCapture(1)

if __name__ == "__main__":

    ball = tennisBall()
    kf = KalmanFilter(dt=1/30, r = 0.1, q =0.2, xvals = 3, ndims = 3)

    cameraMatrix = cameraMatrix()
    f_x,f_y,c_x,c_y  = cameraMatrix.get_values()
    lower_green = ball.get_lower_value()
    higher_green = ball.get_higher_value()
   
    while 1:
        
        # ret, frame = cap.read()
        # if ret is False:
        #          break
        #get RGB, depth, and vison of depth 

        frame = get_video()
        depth = get_depth()
        dvideo = get_dvideo()
  
        blurred_frame = cv2.GaussianBlur(frame, (5,5), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,lower_green,higher_green)
        green_mask = cv2.bitwise_and(frame,frame, mask = mask )
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                M = cv2.moments(contour)
                
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                center2 = (int(M["m01"] / M["m00"]),int(M["m10"] / M["m00"]))
             
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.drawContours(frame, contour, -1, (0,255,0), 3)
                pts.appendleft(center)

                x_world = (center2[0] - c_x) * depth[center2] / f_x
                y_world = (center2[1] - c_y) * depth[center2] / f_y
        


        # cv2.namedWindow('RGB image')        # Create a named window
        # cv2.moveWindow('RGB image', 40,300)  # Move it to (40,30)

        
      

        coord.append([x_world,y_world])
        coord_z.append(int(depth[center2]))
        coord_full.append([int(x_world),int(y_world),int(depth[center2])])
        # print(coord_full[-1])

        # KALMAN
        predictions = calculateKalman(coord_full[-1],kf)
       

        x_screen = (predictions[0]/predictions[2] * f_x + c_x)
        y_screen = (predictions[1]/predictions[2] * f_y + c_y)
        

        # x_screen = (x_world/depth[center2]) * f_x + c_x
        # y_screen = (y_world/depth[center2]) * f_y + c_y

        # print(coord_full[-1])
        # print(int(x_screen),int(y_screen))
        # print(int(x), int(y))

        # print(int(x_world),int(y_world))
        cv2.circle(frame, (int(y_screen),int(x_screen)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(x_screen),int(y_screen)), 5, (255, 100, 0), -1)

        



        cv2.imshow('Depth image',dvideo)
        cv2.imshow('RGB image',frame)




        key = cv2.waitKey(1)
        if cv2.waitKey(33) == ord('a'):
            break
      
    cv2.destroyAllWindows()
        
    runFinal(coord_full)


    # ball.set_xy(coord)
    # ball.set_z(coord_z)
    print(coord_full)
    ball.set_full(coord_full)


    # ball.save_trajectory_xy()
    # ball.save_trajectory_z()
    ball.save_trajectory_full()

    # xs = range(len(coord_z))
    # plt.plot(xs,coord_z)
    # plt.show()
    # print(ball.show_3D_trajectory())

    

