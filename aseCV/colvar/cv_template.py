import pymp
from ase import Atoms
import numpy as np

class CVTemplate:
    def __init__(self,atoms:Atoms,indices:list):
        super().__init__()
        #mandatory stuff for any CV
        self.atoms = atoms # Atoms object
        self.indices = indices # list of indices of the atoms to be used in the CV
        self.cv = 0 # initial value of the CV
        self.dx = pymp.shared.array(len(self.indices)) # derivatives of the MACE distance CV with respect to x-coordinates
        self.dy = pymp.shared.array(len(self.indices)) # derivatives of the MACE distance CV with respect to y-coordinates
        self.dz = pymp.shared.array(len(self.indices)) # derivatives of the MACE distance CV with respect to z-coordinates
    # Calculate the value of your CV
    def get_cv(self):
        return self.cv
    
    # Calculate the value of the derivatives of youtr CV with respect of the coordinates of the involved atoms
    def get_derivatives(self):
        with pymp.Parallel() as p:
            for i in p.range(len(self.indices)):
                self.dx[i] =  0
                self.dy[i] =  0
                self.dz[i] =  0
        return self.dx, self.dy, self.dz
    
    # Calculate the value of the CV and its derivatives
    def cv_calc(self):
        self.get_cv()
        self.get_derivatives()
    
    def print_cv(self):
        self.cv_calc()
        print(f"CV = {self.cv}")
        print(f"CV derivatives, with respect to x-coordinates: {self.dx}")
        print(f"CV derivatives, with respect to y-coordinates: {self.dy}")
        print(f"CV derivatives, with respect to z-coordinates: {self.dz}")
