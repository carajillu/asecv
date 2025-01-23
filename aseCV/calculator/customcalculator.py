from ase.calculators.calculator import Calculator, all_changes

class CustomCalculator(Calculator):
    """CustomCalculator adds biases to a base calculator's energy and forces."""
    implemented_properties = ['energy', 'forces']  # Extend as needed

    def __init__(self, base_calculator, cv=None, bias=None):
        super().__init__()  # Initialize the parent Calculator class
        self.base_calculator = base_calculator
        self.cv = cv
        self.bias = bias

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # Call the base calculator's calculate method
        self.base_calculator.calculate(atoms, properties, system_changes)
        
        if self.cv is None or self.bias is None:
            return

        # Copy results from the base calculator
        self.results = self.base_calculator.results.copy()
        self.cv.cv_calc()
        self.bias.bias_calc()
        # Modify energy with the bias
        if "energy" in self.results:
            self.results["energy"] += self.bias.bias

        # Modify forces with the bias
        if "forces" in self.results:
            for i in range(len(self.cv.indices)):
                self.results["forces"][self.cv.indices[i]] += self.bias.bias_forces_x[i]
                self.results["forces"][self.cv.indices[i]] += self.bias.bias_forces_y[i]
                self.results["forces"][self.cv.indices[i]] += self.bias.bias_forces_z[i]
