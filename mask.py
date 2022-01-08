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

                print(center)
                print(depth[center2])

                x_world = (int(M["m01"] / M["m00"])- c_x) * depth[center2] / f_x
                y_world = (int(M["m10"] / M["m00"]) - c_y) * depth[center2] / f_y
                print(x_world, y_world)
        #
        #  for i in np.arange(1, len(pts)):
        #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        #     #print('done1')
        #     if counter >= 10 and i == 1 and pts[-10] is not None:
        #         #print('done2')
        #         dX = pts[-10][0] - pts[i][0]
        #         dY = pts[-10][1] - pts[i][1]
        #         (dirX, dirY) = ("", "")
        #         cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
        #         ( 10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #         0.35, (0, 0, 255), 1)


        cv2.namedWindow('RGB image')        # Create a named window
        cv2.moveWindow('RGB image', 40,300)  # Move it to (40,30)

        cv2.imshow('RGB image',frame)
        cv2.imshow('Depth image',dvideo)
        # plt.imshow(depth)
      
        #cv2.imshow('Depth image', depth)
        # cv2.imshow('Mask image',mask)
        # cv2.imshow ('Filtracja', green_mask)

        coord.append(center2)
        coord_z.append(int(depth[center2]))
        coord_full.append(center + (int(depth[center2]),))

        key = cv2.waitKey(1)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        k = cv2.waitKey(5) & 0xFF
        if cv2.waitKey(33) == ord('a'):
            break
        if k == 0:
            break
        # #print(pts)
        # counter += 1
        
    cv2.destroyAllWindows()
        
    runDemo(coord_full)


    ball.set_xy(coord)
    ball.set_z(coord_z)
    ball.set_full(coord_full)


    ball.save_trajectory_xy()
    ball.save_trajectory_z()
    ball.save_trajectory_full()

    xs = range(len(coord_z))
    plt.plot(xs,coord_z)
    plt.show()
    print(ball.show_3D_trajectory())

    

