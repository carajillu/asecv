import numpy as np
import pymp
from ase import Atoms


class Harmonic:
    def __init__(self,cv,k,x0):
        self.cv = cv
        self.k = k
        self.x0 = x0
        self.bias = 0.0
        self.bias_forces_x = pymp.shared.array(len(self.cv.indices))
        self.bias_forces_y = pymp.shared.array(len(self.cv.indices))
        self.bias_forces_z = pymp.shared.array(len(self.cv.indices))
    
    def get_bias(self):
        self.cv.cv_calc()
        self.bias = 0.5*self.k*(self.cv.cv - self.x0)**2
        return self.bias
    
    def get_bias_forces(self):
        with pymp.Parallel() as p:
            for i in range(len(self.cv.indices)):
                self.bias_forces_x[i] = -self.k*(self.cv.cv - self.x0)*self.cv.dx[i]
                self.bias_forces_y[i] = -self.k*(self.cv.cv - self.x0)*self.cv.dy[i]
                self.bias_forces_z[i] = -self.k*(self.cv.cv - self.x0)*self.cv.dz[i]
        return self.bias_forces_x, self.bias_forces_y, self.bias_forces_z
    
    def bias_calc(self):
        self.get_bias()
        self.get_bias_forces()
    
    def print_bias(self):
        self.bias_calc()
        print(f"Harmonic bias: {self.bias}")
        print(f"Harmonic bias forces x: {self.bias_forces_x}")
        print(f"Harmonic bias forces y: {self.bias_forces_y}")
        print(f"Harmonic bias forces z: {self.bias_forces_z}")