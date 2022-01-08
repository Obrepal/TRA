class cameraMatrix():

    def __init__(self,f_x = 528.6553,f_y = 529.985, c_x = 315.584, c_y = 269.3093):
        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y
    
    def get_values(self):
        return self.f_x, self.f_y, self.c_x, self.c_y
        