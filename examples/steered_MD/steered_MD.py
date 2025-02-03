import argparse
import pandas as pd
import numpy as np
import wget
import warnings
#warnings.filterwarnings("ignore")
import sys

from ase.io import read, write
from ase.units import fs, kB
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger

from mace.calculators import mace_off

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
    parser.add_argument("-t","--temperature", type=float, help="Temperature in K",default=300)
    parser.add_argument("-w","--windows", type=int, help="Number of bias windows (run continuously as steered MD)",default=1)
    parser.add_argument("-n","--nsteps", type=int, help="Number of steps per window",default=1000)
    parser.add_argument("-m","--model", type=str, help="MACE model to use as force field",default=None)
    parser.add_argument("-d","--device", default="cpu", type=str, help="Device to use for the MACE model")
    parser.add_argument("--nowarnings", action="store_true", help="Suppress warnings")
    return parser.parse_args()
    

def check_systems(atoms,atoms_ref):
    assert len(atoms)==len(atoms_ref),"Different number of atoms in the two systems"
    for i in range(len(atoms)):
        assert atoms[i].symbol==atoms_ref[i].symbol,f"Element mismatch at atom {i}: {atoms[i].symbol} vs {atoms_ref[i].symbol}"
    print("Systems are compatible")
    return

def return_print_snapshot(atoms,basename):
    def print_snapshot():
        #atoms.arrays["forces"]=atoms.get_forces()
        atoms.arrays['node_energy']=atoms.calc.results['node_energy']
        atoms.center()
        write(f"{basename}_trj.xyz",atoms,format="extxyz",append=True)
    return print_snapshot

def return_print_log(dyn):
    def print_log():
        print(f"Time: {dyn.get_time()/(1000 * fs)} ps, CV: {dyn.atoms.cv.value}")
    return print_log

if __name__=="__main__":
    args = parse()
    print(args)

    if args.nowarnings:
        warnings.filterwarnings("ignore")

    atoms=read(args.input_xyz)
    atoms.cell=[10,10,10]
    atoms.center()
    atoms.pbc=True
    atoms_ref=read(args.ref_xyz)
    check_systems(atoms,atoms_ref)

    # Base calculator
    calc = mace_off(model="small",device=args.device)

    # Get Reference energies
    results_ref=calc.calculate(atoms_ref)
    e0s = [calc.results["node_energy"][i] for i in args.atoms_idx]

    #define the CV
    atoms.cv=MACE_dist(atoms=atoms,calculator=calc,indices=args.atoms_idx,e0s=e0s)
    atoms.cv.cv_calc()
    r_init=atoms.cv.value
    print(f"initial value of the MACE distance CV: {r_init}")


    print("Setting up the dynamics")
    MaxwellBoltzmannDistribution(atoms=atoms, temperature_K=args.temperature)

    print("Running the simulation")
    dr=(args.r0-1.22)/args.windows
    r_windows=[1.22+dr*i for i in range(args.windows+1)]
    for i in range(args.windows+1):
        r_i=r_windows[i]
        print(f"Window {i+1}/{args.windows}: r_i={r_i}")
        print("setting bias")
        bias=Harmonic(cv=atoms.cv,k=args.kappa,x0=r_i)
        print("setting custom calculator")
        atoms.calc=CustomCalculator(base_calculator=calc, cv=atoms.cv, bias=bias)
        print("creating dynamics object")
        dyn=Langevin(atoms=atoms, timestep=1*fs, temperature_K=300, friction=0.01)
        Logger=MDLogger(dyn=dyn,atoms=atoms, logfile="log.txt", header=True, stress=False, peratom=False, mode="w")
        dyn.attach(Logger, interval=100)
        print_snapshot=return_print_snapshot(atoms,args.input_xyz.split(".")[0])
        dyn.attach(print_snapshot, interval=100)
        print_log=return_print_log(dyn)
        dyn.attach(print_log, interval=100)
        print("running dynamics")
        dyn.run(args.nsteps)



    
    
    
    

    






