import argparse
import pandas as pd
import numpy as np
import wget
import warnings

from ase.io import read, write
from ase.units import fs, kB
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mace.calculators import MACECalculator

from aseCV.colvar.mace_dist import MACE_dist
from aseCV.colvar.cv_template import CVTemplate
from aseCV.bias.harmonic import Harmonic
from aseCV.calculator.customcalculator import CustomCalculator



def parse():
    parser = argparse.ArgumentParser(description="Converts a list of numbers to a list of ranges")
    parser.add_argument("-i", "--input_xyz", required=True, type=str, help="Input xyz file")
    parser.add_argument("-r","--ref_xyz", required=True, type=str, help="Reference xyz file")
    parser.add_argument("-b", "--backward", action="store_true", help="Print PMF in reverse")
    parser.add_argument("-a","--atoms_idx", required=True, type=int, nargs="+", help="Atoms to include in the bias")
    parser.add_argument("-r0","--r0", required=True, type=float, help="Reference distance in the atomic energy space")
    parser.add_argument("-k","--kappa", required=True, type=float, help="Force constant in the atomic energy space")
    parser.add_argument("-m","--model", required=True, type=str, help="MACE model to use as force field")
    parser.add_argument("-d","--device", default="cpu", type=str, help="Device to use for the MACE model")
    parser.add_argument("--nowarnings", action="store_true", help="Suppress warnings")
    return parser.parse_args()
    

def check_systems(atoms,atoms_ref):
    assert len(atoms)==len(atoms_ref),"Different number of atoms in the two systems"
    for i in range(len(atoms)):
        assert atoms[i].symbol==atoms_ref[i].symbol,f"Element mismatch at atom {i}: {atoms[i].symbol} vs {atoms_ref[i].symbol}"
    print("Systems are compatible")
    return

if __name__=="__main__":
    args = parse()
    print(args)

    if args.nowarnings:
        warnings.filterwarnings("ignore")

    atoms=read(args.input_xyz)
    atoms_ref=read(args.ref_xyz)
    check_systems(atoms,atoms_ref)

    if args.model.startswith("https://"):
        args.model = wget.download(args.model)
    calc = MACECalculator(model_paths=args.model,device=args.device)
    
    # Get system MACE forces for debugging
    atoms.calc = calc
    atoms.get_potential_energy()
    print(f"Energy: {atoms.calc.results['energy']}")
    print(f"Forces: {atoms.calc.results['forces'][args.atoms_idx]}")
    #Get reference MACE energies
    atoms_ref.calc = calc
    atoms_ref.get_potential_energy()
    e0s = atoms_ref.calc.results["node_energy"][args.atoms_idx]
    print(f"Reference Energies: {e0s}")

    #define colvar
    cv = MACE_dist(atoms=atoms,calculator=calc,indices=args.atoms_idx,e0s=e0s)
    cv.print_cv()

    #define bias
    bias=Harmonic(cv,k=args.kappa,x0=args.r0)
    bias.print_bias()

    #define calculator
    customcalc=CustomCalculator(base_calculator=calc,cv=cv,bias=bias)
    atoms.calc = customcalc
    atoms.get_potential_energy()
    print(f"Energy: {atoms.calc.results['energy']}")
    print(f"Forces: {atoms.calc.results['forces'][args.atoms_idx]}")

    

    






