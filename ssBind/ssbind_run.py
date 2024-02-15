#!/usr/bin/python
import os, math, glob, uuid, shutil, csv
import pandas as pd
import argparse
import itertools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import * 
import multiprocessing as mp
from spyrmsd import io, rmsd

#Substructure-based alternative BINDing modes generator for protein-ligand systems

	
def ParserOptions():
    parser = argparse.ArgumentParser()

    """Parse command line arguments."""
    parser.add_argument("--reference", dest="reference", help="Referance molecule", required=True)   
    parser.add_argument("--ligand", dest="ligand", help="Ligand molecule", required=True)
    parser.add_argument("--receptor", dest="receptor", help="PDB file for receptor protein", required=True)
    parser.add_argument("--FF", dest="FF", default='gaff', help="Generalized Force Fields GAFF, CGenFF, OpenFF", choices=['gaff', 'openff', 'cgenff'])    
    parser.add_argument("--proteinFF", dest="proteinFF", default='amber99sb-ildn')    
    parser.add_argument("--degree", dest="degree", type=float,help="Amount, in degrees, to enumerate torsions by (default 60.0)", default=60.0) 
    parser.add_argument("--cutoff", dest="cutoff_dist", type=float,help="Cutoff for eliminating any conformer close to protein within cutoff by (default 1.5 A)", default=1.5) 
    parser.add_argument("--rms", dest="rms", type=float,help="Only keep structures with RMS > CUTOFF (default 0.2 A)", default=0.2) 
    parser.add_argument("--cpu", dest="cpu", type=int, help="Number of CPU. If not set, it uses all available CPUs.") 
    parser.add_argument("--generator", dest="generator", help="Choose a method for the conformer generation.", choices=['angle', 'rdkit', 'plants', 'rdock']) 
    parser.add_argument("--numconf", dest="numconf", type=int, help="Number of confermers", default=1000)    
    parser.add_argument("--minimize", dest="minimize", help="Perform minimization", choices=['gromacs', 'smina'])    
    parser.add_argument("--flex", dest="flex", help="Residues having side-chain flexibility taken into account")
    parser.add_argument("--bin", dest="bin", type=float, help="Numeric vector giving bin width in both vertical and horizontal directions in PCA analysis", default=0.25)
    parser.add_argument("--distThresh", dest="distThresh", type=float, help="elements within this range of each other are considered to be neighbors during clustering", default=0.5)
    parser.add_argument("--numbin", dest="numbin", type=int, help="Number og bins to be extract for clustering conformations", default=10)
    args = parser.parse_args()
    return args

def close_pool():
    global pool
    pool.close()
    pool.terminate()
    pool.join()    

def main():

	from chem_tools import MolFromInput, get_uniqueDihedrals, degreeRange
		
	args = ParserOptions()

	if args.cpu is not None:
		nprocs = args.cpu
	else:
		nprocs = mp.cpu_count()
		
	pool = mp.Pool(processes=nprocs)
	terminate_signal = mp.Event()
	
	refmol = MolFromInput(os.path.abspath(args.reference))
	input_file = MolFromInput(os.path.abspath(args.ligand))
	
	
	molDihedrals = get_uniqueDihedrals(refmol, input_file)
	inputs = itertools.product(degreeRange(args.degree),repeat=len(molDihedrals))
	
	if args.generator == 'angle':
		from chem_tools import gen_conf_angle
		#Conformer generation based on dihedral angle enumeration.
		print(f"\nNumber of CPU cores in use for conformer generation: {nprocs}")
		if(len(molDihedrals) > 3):
			print("\nWarning! Too many torsions ({})".format(len(molDihedrals)))
		
		if(len(molDihedrals) > 5):
		    close_pool()
		    exit()
		pool.starmap(gen_conf_angle, [(j, molDihedrals, input_file, refmol) for j in inputs])
		
		print('\nConformational sampling is running for {} dihedrals.'.format(len(molDihedrals)))
	
	elif args.generator == 'plants':
		from chem_tools import clustering_poses
		import plants

		#Conformer generation using PLANTS docking tool.
		print(f"\nNumber of CPU cores in use for conformer generation using PLANTS: {nprocs}")
		
		#Get docking center
		xyz = " ".join(map(str, [round(coord, 3) for coord in plants.molecule_center(input_file)]))
		
		#Get the atom index to be constrainted
		fixedAtom = plants.getAtomConst(input_file, refmol)
		
		if args.flex:
			flex_res = [f"flexible_protein_side_chain_string {residue[0]}{residue[1]}" for residue in plants.get_flex_residues(receptor, ligand, cutoff=4)]
		else:
			flex_res = []

		plants.SPORES(args.receptor, 'receptor.mol2', 'settypes')

		pool.starmap(plants.plants_docking, [(i, input_file, refmol, args.receptor, 12, f'output_{i}', 10, xyz, fixedAtom+1, flex_res) for i in range(math.ceil(args.numconf/10))])	
		mol2_files = glob.glob(os.path.join('docking_conformers', "*.mol2"))
		print('\n{} conformers have been generated using PLANTS docking tool.'.format(len(mol2_files)))
		#pool.starmap(plants.filtering, [(MolFromInput(mol), args.numconf, args.rms) for mol in mol2_files])
		plants.combine_files('docking_conformers')
		clustering_poses('conformers.sdf', input_file, 'Scores.csv', 'out.svg', binsize=args.bin, distThresh=args.distThresh, numbin=args.numbin)
	elif args.generator == 'rdock':
		import rdock
		from chem_tools import clustering_poses
		
		rdock_random = str(uuid.uuid4())
		os.makedirs('.{}'.format(rdock_random))
		
		
		molecule = os.path.abspath(args.reference)
		
		input_format = molecule.split('.')[-1].lower()

		if input_format not in {'sd','sdf'}:
			obabel_convert(molecule, 'ref_{}.sdf'.format(rdock_random), QniqueNames=False)
			molecule = 'ref_{}.sdf'.format(rdock_random)
		
		rdock.get_tethered(refmol, input_file, rdock_random)
		rdock.prepare_receptor(RECEPTOR_FILE = os.path.abspath(args.receptor), REF_MOL = molecule)
		pool.starmap(rdock.run_rdock, [(i, f'.{rdock_random}/{rdock_random}.sd', f'{rdock_random}') for i in range(math.ceil(args.numconf/10))])
		rdock.combine_files(f'.{rdock_random}')
		shutil.rmtree(f'.{rdock_random}')
		clustering_poses('conformers.sdf', input_file, 'Scores.csv', 'out.svg', binsize=args.bin, distThresh=args.distThresh, numbin=args.numbin)
	else:
		from chem_tools import gen_conf_rdkit
		#Conformer generation using RDKit.
		print(f"\nNumber of CPU cores in use for conformer generation: {nprocs}")
		if args.numconf:
			pool.starmap(gen_conf_rdkit, [(input_file, refmol, j) for j in range(args.numconf)])
		else:
			pool.starmap(gen_conf_rdkit, [(input_file, refmol, j) for j in range(len(list(inputs)))])
	
	if args.generator in ('rdkit', 'angle'):
		###Filter conformers having stearic clashes, clash with the protein, duplicates.
		from chem_tools import filtering
		print('\n{} conformers have been generated.'.format(len(Chem.SDMolSupplier('conformers.sdf'))))
		pool.starmap(filtering, [(mol, args.receptor, args.cutoff_dist, args.rms) for i, mol in enumerate(Chem.SDMolSupplier('conformers.sdf'))])
	


if __name__ == '__main__':
	
	from chem_tools import *
	from plotting import *
	args = ParserOptions()
	

	if args.cpu is not None:
		nprocs = args.cpu
	else:
		nprocs = mp.cpu_count()
		
	pool = mp.Pool(processes=nprocs)

	with pool:
		main()	
        
	
	if args.minimize == 'gromacs':
		from gmx_tools import *
		
		### Simply optimize the molecule before FF generation
		sdf = MolFromInput('filtered.sdf')
		writer = Chem.SDWriter("ligand.sdf")
		mol = Chem.AddHs(sdf, addCoords=True)
		AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
		writer.write(mol)
		writer.close()
		
		## Getting md setup files
		system_setup(args.receptor, 'ligand.sdf',  proteinFF=args.proteinFF, FF=args.FF)
		
		## Making directory for trajectory files
		trjdir = str(uuid.uuid4())
		os.makedirs('.{}'.format(trjdir))
		
		## Running minimization in parallel
		pool.starmap(minimize, [(i, mol, trjdir) for i, mol in enumerate(Chem.SDMolSupplier('filtered.sdf'))])
		
		## Combining trajectories 
		cmd = '{0} trjcat -f {1} -o trjout.xtc  > gromacs.log 2>&1'.format(find_gmx(), f".{trjdir}/*")
		RC = os.system(cmd)
		if RC != 0:
			raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("gromacs.log\n"))

		with open('Scores.csv', 'r') as unsorted_csv_file:
			csv_reader = csv.reader(unsorted_csv_file)
			sorted_csv_rows = sorted(csv_reader, key=lambda x: int(x[0]))

		with open('Scores.csv', 'w', newline='') as sorted_csv_file:
			csv_writer = csv.writer(sorted_csv_file)
			csv_writer.writerow(['Index', 'Score'])
			csv_writer.writerows(sorted_csv_rows)
		## Deleting temperory trajectory files
		shutil.rmtree(f".{trjdir}")
		
		clustering_poses('trjout.xtc', MolFromInput(args.ligand), 'Scores.csv', 'out.svg', binsize=args.bin, distThresh=args.distThresh, numbin=args.numbin)
		
	elif args.minimize == 'smina':
	
		import smina
		from chem_tools import clustering_poses
		pool.starmap(smina.smina_minimize_score, [(i, args.receptor, mol) for i, mol in enumerate(Chem.SDMolSupplier('filtered.sdf'))])
    
		
		smina.combine_sdf_files('minimized_conformers.sdf', 'minimized_conformers', 'Scores.csv')

		clustering_poses('minimized_conformers.sdf', MolFromInput(args.ligand), 'Scores.csv', 'out.svg', binsize=args.bin, distThresh=args.distThresh, numbin=args.numbin)


	close_pool()

