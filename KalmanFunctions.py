import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from KalmanFilter import KalmanFilter


def plot3D(xMeas, yMeas, zMeas, xKalman, yKalman, zKalman,):
  
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.scatter(xKalman, yKalman, zKalman,color='green',label='Kalman')
    ax.scatter(xMeas, yMeas, zMeas, color='red',label= 'Measurments')
    ax.legend()
    plt.show()


# def calculateKalman(measurements):
#     kf = KalmanFilter(dt=1/30, r = 0.1, q =0.2, xvals = 3, ndims = 3)
#     results = []
#     A = kf.get_A()
#     print(A)

#     for measurement in measurements:
#         measArray = np.asarray([[measurement[0]],[measurement[1]],[measurement[2]]])
#         x = kf.run(measArray)
#         results.append((x[:3]).tolist())

#     return results

def calculateKalman(measurement,kf):
    
    measArray = np.asarray([[measurement[0]],[measurement[1]],[measurement[2]]])
    x = kf.run(measArray)
    return x[:3]

def show_3D(measurements,kf):
    results = []
    for measurement in measurements:
        measArray = np.asarray([[measurement[0]],[measurement[1]],[measurement[2]]])
        x = kf.run(measArray)
        results.append((x[:3]).tolist())

    return results

def runFinal(measurements):
    kf = KalmanFilter(dt=1/30, r = 0.1, q =0.2, xvals = 3, ndims = 3)
    results = show_3D(measurements,kf)

    xMeas, yMeas, zMeas = list(zip(*measurements))
    xKalman, yKalman, zKalman = list(zip(*results))
    
    plot3D(xMeas, yMeas, zMeas, xKalman, yKalman, zKalman)
    

    

