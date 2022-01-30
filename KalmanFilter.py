import numpy as np
import matplotlib.pyplot as plt
class KalmanFilter:

    def __init__(self, dt=1/30, ndims = 3, xvals = 3, zdims = None, pu = 100, A = np.array([None]), r=1, q=0.5):
     
        self.dt = float(dt) 
        self.ddt = (self.dt**2)/2

        xdims = ndims * xvals
        xsize = (xdims,1)

        #previous state vector
        self.x = np.zeros(xsize) 

        #covariance matrix
        self.P = np.eye(xdims) * pu 

        self.A = np.array([[1, 0, 0 ,self.dt, 0, 0, self.ddt,0,0],
                           [0, 1, 0, 0, self.dt, 0, 0,self.ddt,0],
                           [0, 0, 1, 0, 0,self. dt, 0, 0,self.ddt],
                           [0, 0, 0, 1, 0, 0, self.dt,0, 0],
                           [0, 0, 0, 0, 1, 0, 0,self.dt, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, self.dt],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1],
                          ])

        self.Q = np.eye(self.x.shape[0]) * q # process noise matrix

        self.B = np.eye(self.x.shape[0])

        self.u = np.zeros((self.x.shape[0],1))

        if not zdims:
            zdims = ndims

        zsize = (zdims,1)
        self.z = np.zeros(zsize)

        self.H = np.zeros((zdims,xdims))

        for i in range(0,ndims):
            self.H[i][i] = 1

        self.R = np.eye(self.z.shape[0]) * r

        self.lastResult = np.zeros(xsize)

    def predict(self):
           

          self.x = np.dot(self.A, self.x)
          self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

          self.lastResult = self.x
          return self.x
          """Predict state vector, u and variance of uncertainty (covariance), P.
                where,
                x: previous state estimate
                P: previous covariance matrix (k-1)
                A: state transition (nxn) matrix (k-1)
                Q: process noise covariance matrix
                B: input effect matrix
                u: control input
            Equations:
                X_{k} = A * x_{k-1} + B * u_{k}
                P_{k} = A * P_{k-1} * A.T + Q
                where,
                    A.T is F transpose
            Args:
                None
            Return:
                vector of predicted state estimate, X

            Save to member variables:
                vector of predicted state estimate, X
                vector of covariance, P
            """

    def update(self, z):
          self.z = z

          C = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
          K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(C)))
          V = self.z - np.dot(self.H, self.x)

          self.x = self.x + np.dot(K, V)

          self.P = self.P - np.dot(K, np.dot(C, K.T))

          self.lastResult = self.x

          return self.x
          

    def run(self,z):

            xPredicted = self.predict()

            xUpdated = self.update(z)

            return xUpdated

    def get_A(self):
        return self.A


