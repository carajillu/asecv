import argparse
import pandas as pd
import numpy as np

from ase.io import read, write
from ase.units import fs, kB
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mace.calculators import MACECalculator



def parse():
    parser = argparse.ArgumentParser(description="Converts a list of numbers to a list of ranges")
    parser.add_argument("-i", "--input_xyz", required=True, type=str, help="Input xyz file")
    parser.add_argument("-r","--ref_xyz", required=True, type=str, help="Reference xyz file")
    parser.add_argument("-b", "--backward", action="store_true", help="Print PMF in reverse")
    parser.add_argument("-a","--atoms", required=True, type=int, nargs="+", help="Atoms to include in the bias")
    parser.add_argument("-r0","--r0", required=True, type=float, help="Reference distance in the atomic energy space")
    parser.add_argument("-k","--kappa", required=True, type=float, help="Force constant in the atomic energy space")
    parser.add_argument("-m","--model", required=True, type=str, help="MACE model to use as force field")
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

    atoms=read(args.input_xyz)
    atoms_ref=read(args.ref_xyz)
    check_systems(atoms,atoms_ref)

