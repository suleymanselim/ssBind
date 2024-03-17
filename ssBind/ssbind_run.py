#!/usr/bin/python
from ssBind import SSBIND
from ssBind.chem_tools import MolFromInput
import argparse
import os
import multiprocessing as mp

#Substructure-based alternative BINDing modes generator for protein-ligand systems


def ParserOptions():
    parser = argparse.ArgumentParser()

    """Parse command line arguments."""
    parser.add_argument("--reference", dest="reference", help="Referance molecule", required=True)   
    parser.add_argument("--ligand", dest="ligand", help="Ligand molecule", required=True)
    parser.add_argument("--receptor", dest="receptor", help="PDB file for receptor protein", required=True)
    parser.add_argument("--FF", dest="FF", default='gaff', help="Generalized Force Fields GAFF, CGenFF, OpenFF", choices=['gaff', 'gaff2', 'openff', 'cgenff'])    
    parser.add_argument("--proteinFF", dest="proteinFF", default='amber99sb-ildn')    
    parser.add_argument("--degree", dest="degree", type=float,help="Amount, in degrees, to enumerate torsions by (default 60.0)", default=60.0) 
    parser.add_argument("--cutoff", dest="cutoff_dist", type=float,help="Cutoff for eliminating any conformer close to protein within cutoff by (default 1.5 A)", default=1.5) 
    parser.add_argument("--rms", dest="rms", type=float,help="Only keep structures with RMS > CUTOFF (default 0.2 A)", default=0.2) 
    parser.add_argument("--cpu", dest="cpu", type=int, help="Number of CPU. If not set, it uses all available CPUs.") 
    parser.add_argument("--generator", dest="generator", help="Choose a method for the conformer generation.", choices=['angle', 'rdkit', 'plants', 'rdock']) 
    parser.add_argument("--numconf", dest="numconf", type=int, help="Number of confermers", default=1000)    
    parser.add_argument("--minimize", dest="minimize", help="Perform minimization", choices=['gromacs', 'smina'])    
    parser.add_argument("--flexDist", dest="flexDist", type=int, help="Residues having side-chain flexibility taken into account. Take an interger to calculate closest residues around the ligand")
    parser.add_argument("--flexList", dest="flexList", type=str, help="Residues having side-chain flexibility taken into account. Take a list of residues for flexibility")
    parser.add_argument("--bin", dest="bin", type=float, help="Numeric vector giving bin width in both vertical and horizontal directions in PCA analysis", default=0.25)
    parser.add_argument("--distThresh", dest="distThresh", type=float, help="elements within this range of each other are considered to be neighbors during clustering", default=0.5)
    parser.add_argument("--numbin", dest="numbin", type=int, help="Number of bins to be extract for clustering conformations", default=10)
    args = parser.parse_args()
    return args



def main(args, nprocs):

    reference_substructure = MolFromInput(args.reference)
    query_molecule = MolFromInput(args.ligand)
    
    conformation_generator = SSBIND(reference_substructure = reference_substructure,
        					query_molecule = query_molecule,
        					receptor_file = args.receptor, nprocs = nprocs, numconf = args.numconf)
    
    if args.generator == 'plants':
        conformation_generator.generate_conformers_plants(flexDist = args.flexDist, flexList = args.flexList)
    elif args.generator == 'rdock':
        receptor_extension = os.path.splitext(args.receptor)[1].lower()
        if receptor_extension != ".mol2":
            print(f"""Warning: {args.receptor} is not a .mol2 file.
            The receptor “.mol2″ file must be preparated (protonated, charged, etc.)""")
        conformation_generator.generate_conformers_rdock()
    else:
        conformation_generator.generate_conformers(generator = args.generator, degree = args.degree, cutoff_dist = args.cutoff_dist, rms = args.rms)

    if args.minimize is not None:
        conformers = 'filtered.sdf' if args.generator in ['rdkit', 'angle'] else 'conformers.sdf'
        conformation_generator.run_minimization(conformers = conformers, minimizer = args.minimize, proteinFF = args.proteinFF, FF = args.FF)
    
    conformers_map = {
    'smina': 'minimized_conformers.sdf',
    'gromacs': 'trjout.xtc'
    }
    conformers = conformers_map.get(args.minimize, 'conformers.sdf')
    conformation_generator.clustering(
                   conformers = conformers, 
                   scores = 'Scores.csv', 
                   binsize = args.bin, 
                   distThresh = args.distThresh, 
                   numbin = args.numbin
                   )

if __name__ == '__main__':
    
    
    args = ParserOptions()
    
    nprocs = args.cpu if args.cpu is not None else mp.cpu_count()
    
    print(f"\nNumber of CPU in use for conformer generation: {nprocs}")
        
    main(args, nprocs)

