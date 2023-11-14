#!/usr/bin/python
import os, math, glob, uuid, shutil
import pandas as pd
import argparse
import itertools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import * 
import multiprocessing as mp

#Substructure-based ALTernative BINDing modes generator for protein-ligand systems


def ParserOptions():
    parser = argparse.ArgumentParser()

    """Parse command line arguments."""
    parser.add_argument("--reference", dest="reference", help="Referance molecule", required=True)   
    parser.add_argument("--ligand", dest="ligand", help="Ligand molecule", required=True)
    parser.add_argument("--receptor", dest="receptor", help="PDB file for receptor protein", required=True)
    parser.add_argument("--FF", dest="FF", default='gaff', help="Generalized Force Fields GAFF, CGenFF, OpenFF", choices=['gaff', 'openff', 'cgenff'])    
    parser.add_argument("--degree", dest="degree", type=float,help="Amount, in degrees, to enumerate torsions by (default 15.0)", default=60.0) 
    parser.add_argument("--cutoff", dest="cutoff_dist", type=float,help="Cutoff for eliminating any conformer close to protein within cutoff by (default 1.5 A)", default=1.5) 
    parser.add_argument("--rms", dest="rms", type=float,help="Only keep structures with RMS > CUTOFF (default 0.2 A)", default=0.2) 
    parser.add_argument("--output", dest="output", help="Output sdf file for conformations", default='output.sdf')
    parser.add_argument("--cpu", dest="cpu", type=int, help="Number of CPU. If not set, it uses all available CPUs.") 
    parser.add_argument("--generator", dest="generator", help="Conformer generator: rdkit or angle enumeration.", choices=['angle', 'rdkit', 'docking']) 
    parser.add_argument("--numconf", dest="numconf", type=int, help="Number of confermers", default=1000)    
    parser.add_argument("--minimize", dest="minimize", help="Perform minimization", action='store_true')
    parser.add_argument("--flex", dest="flex", help="Residues having side-chain flexibility taken into account")
    args = parser.parse_args()
    return args
    

def main():

	from chem_tools import MolFromInput, get_uniqueDihedrals, degreeRange
		
	args = ParserOptions()
	
	refmol = MolFromInput(args.reference)
	input_file = MolFromInput(args.ligand)
	
	
	if args.cpu is not None:
		nprocs = args.cpu
	else:
		nprocs = mp.cpu_count()
	
	pool = mp.Pool(processes=nprocs)
	

	molDihedrals = get_uniqueDihedrals(refmol, input_file)
	inputs = itertools.product(degreeRange(args.degree),repeat=len(molDihedrals))
	
	if args.generator == 'angle':
		from chem_tools import gen_conf_angle
		#Conformer generation based on dihedral angle enumeration.
		print(f"\nNumber of CPU cores in use for conformer generation: {nprocs}")
		if(len(molDihedrals) > 3):
			print("\nWarning! Too many torsions ({})".format(len(molDihedrals)))
		pool.starmap(gen_conf_angle, [(j, molDihedrals, input_file, refmol, args.receptor, args.cutoff_dist, args.output, args.rms) for j in inputs])
		
		print('\nConformational sampling is running for {} dihedrals.'.format(len(molDihedrals)))
	
	elif args.generator == 'docking':
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
		
		plants.SPORES(args.receptor, 'receptor.mol2')
		pool.starmap(plants.plants_docking, [(i, input_file, refmol, args.receptor, 10, 'output_{}'.format(i), 10, xyz, fixedAtom, flex_res) for i in range(math.ceil(args.numconf/10))])	
		
		mol2_files = glob.glob(os.path.join('docking_conformers', "*.mol2"))
		print('\n{} conformers have been generated using PLANTS docking tool.'.format(len(mol2_files)))
		pool.starmap(plants.filtering, [(MolFromInput(mol), args.rms) for mol in mol2_files])
	else:
		from chem_tools import gen_conf_rdkit
		#Conformer generation using RDKit.
		print(f"\nNumber of CPU cores in use for conformer generation: {nprocs}")
		if args.numconf:
			pool.starmap(gen_conf_rdkit, [(input_file, refmol, j) for j in range(args.numconf)])
		else:
			pool.starmap(gen_conf_rdkit, [(input_file, refmol, j) for j in range(len(list(inputs)))])


	
	if args.generator != 'docking':
		###Filter conformers having stearic clashes, clash with the protein, duplicates.
		from chem_tools import filtering
		print('\n{} conformers have been generated.'.format(len(Chem.SDMolSupplier('conformers.sdf'))))
		pool.starmap(filtering, [(mol, args.receptor, args.cutoff_dist, args.rms) for i, mol in enumerate(Chem.SDMolSupplier('conformers.sdf'))])


if __name__ == '__main__':
	
	from chem_tools import MolFromInput, obabel_convert
	curdir = os.getcwd()
	args = ParserOptions()
	
	#main()	
	
	if args.minimize:
		from gmx_tools import system_setup, combine_gro_files, gmx_mdrun
		
		### Optimize the molecule before FF generation
		sdf = MolFromInput('filtered.sdf')
		writer = Chem.SDWriter("ligand.sdf")
		mol = Chem.AddHs(sdf, addCoords=True)
		AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
		writer.write(mol)
		writer.close()
		
		
		if args.FF == 'cgenff':
			obabel_convert("ligand.sdf", 'ligand.mol2')
			system_setup(args.receptor, 'ligand.mol2',  FF=args.FF)
		else:
			system_setup(args.receptor, 'ligand.sdf',  FF=args.FF)
		
		poses = Chem.SDMolSupplier('filtered.sdf')
		for i, mol in enumerate(poses):
			writer = Chem.SDWriter("ligand.sdf")
			mol = Chem.AddHs(mol, addCoords=True)
			writer.write(mol)
			writer.close()
			
			obabel_convert("ligand.sdf", f'{i}.gro')
			
			with open("md_setup/LIG.gro") as file1, open(f'{i}.gro') as file2:
				for line1, line2 in zip(file1, file2):
					
					
					
			md_dir = str(uuid.uuid4())
			shutil.copytree('md_setup', md_dir)
			
			combine_gro_files('md_setup/protein.gro', 'LIG.gro', md_dir + '/complex.gro')
			os.chdir(md_dir)
			os.system("echo 'q'|gmx make_ndx -f complex.gro")
			shutil.copy('/home/suleymanselim/PROJECTS/saltbind/saltfind/utils/em.mdp', 'em.mdp')
			gmx_mdrun() 
			if i == 3:
				exit()












