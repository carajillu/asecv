import pymp
from ase import Atoms
from mace.calculators import MACECalculator
import numpy as np
class MACE_dist(Atoms):
    def __init__(self,atoms:Atoms,calculator:MACECalculator,indices:list,e0s:list):
        super().__init__()
        #mandatory stuff for any CV
        self.atoms = atoms # atoms object
        self.indices = indices # list of indices of the atoms to be used in the CV
        self.cv = 0 # MACE distance CV
        self.dx = pymp.shared.array(len(self.indices)) # derivatives of the MACE distance CV with respect to x-coordinates
        self.dy = pymp.shared.array(len(self.indices)) # derivatives of the MACE distance CV with respect to y-coordinates
        self.dz = pymp.shared.array(len(self.indices)) # derivatives of the MACE distance CV with respect to z-coordinates
        #specific stuff for this CV
        self.e0s = e0s # list of reference atomic energy values
        self.ediffs = pymp.shared.array(len(self.indices)) # list of per-atom energy differences
        self.atoms.calc = calculator
        assert isinstance(self.atoms.calc, MACECalculator), "This CV can only be used with MACE calculators."
        self.atoms.get_potential_energy() # run the calculator when you init the CV
    
    def get_ediffs(self):
        with pymp.Parallel() as p:
            for i in p.range(len(self.indices)):
                self.ediffs[i] = self.atoms.calc.results["node_energy"][self.indices[i]] - self.e0s[i]
        return self.ediffs
    
    def get_cv(self):
        self.cv = np.sqrt(np.sum(self.ediffs**2))
        return self.cv
    
    def get_derivatives(self):
        with pymp.Parallel() as p:
            for i in p.range(len(self.indices)):
                self.dx[i] =  -self.ediffs[i]/self.cv*self.atoms.calc.results["forces"][self.indices[i]][0]
                self.dy[i] =  -self.ediffs[i]/self.cv*self.atoms.calc.results["forces"][self.indices[i]][1]
                self.dz[i] =  -self.ediffs[i]/self.cv*self.atoms.calc.results["forces"][self.indices[i]][2]
        return self.dx, self.dy, self.dz
    
    def cv_calc(self):
        self.get_ediffs()
        self.get_cv()
        self.get_derivatives()
    
    def print_cv(self):
        self.cv_calc()
        print(f"MACE distance CV: {self.cv}")
        print(f"MACE distance CV derivatives, with respect to x-coordinates: {self.dx}")
        print(f"MACE distance CV derivatives, with respect to y-coordinates: {self.dy}")
        print(f"MACE distance CV derivatives, with respect to z-coordinates: {self.dz}")
