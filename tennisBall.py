import numpy as np
import json
import matplotlib.pyplot as plt

class tennisBall:
    def __init__(self, lower_value = np.array([25, 52, 72]) , higer_value = np.array([102, 255, 255]), coord_xy = 0,coord_z = 0, coord_full = 0):
        self.lower_value = lower_value
        self.higher_value = higer_value
        self.coord_xy = coord_xy
        self.coord_z = coord_z
        self.coord_full = coord_full

    
    def save_trajectory_xy(self):
         with open('Materialy/coord_xy.txt', 'w') as f:
            f.write(json.dumps(self.coord_xy))

    def save_trajectory_z(self):
         with open('Materialy/coord_z.txt', 'w') as f:
            f.write(json.dumps(self.coord_z))

    def save_trajectory_full(self):
         with open('Materialy/coord_full.txt', 'w') as f:
            f.write(json.dumps(self.coord_full))
            

    def show_3D_trajectory(self):
        x, y, z  = zip( * self.coord_full)
        ax = plt.axes(projection="3d")
        ax.plot(x, y, z)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        
        plt.show()
         


    #setters
    def set_xy(self, coord_xy):
        self.coord_xy = coord_xy

    def set_z(self, coord_z):
        self.coord_z = coord_z

    def set_full(self, coord_full):
        self.coord_full = coord_full





    
    #getters
    def get_lower_value(self):
        return self.lower_value
    
    def get_higher_value(self):
        return self.higher_value
    


#higher_green = np.array([50, 255, 255])
#Próba moja z aplikacji
#lower_green = np.array([19, 71, 49])
#higher_green = np.array([61, 209, 255])
            