class Harmonic:
    def __init__(self,cv,k,x0):
        self.cv = cv
        self.k = k
        self.x0 = x0
    
    def get_bias(self):
        return 0.5*self.k*(self.cv - self.x0)**2
    
    def get_bias_derivatives(self):
        return self.k*(self.cv - self.x0)