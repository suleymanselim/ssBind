import glob
import os
import re
import shutil
import sys
import uuid
import warnings
from pathlib import Path

import pandas as pd
import MDAnalysis as mda
from MDAnalysisTests.datafiles import AUX_EDR
from rdkit import Chem
from openff.interchange import Interchange
from openff.toolkit.topology import Topology, Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from acpype.topol import AbstractTopol, ACTopol, MolTopol, header



warnings.simplefilter("ignore")



def find_gmx():
	RC = os.system('gmx -h >/dev/null 2>&1')
	if RC == 0:
		return 'gmx'
	RC = os.system('gmx_mpi -h >/dev/null 2>&1')
	if RC == 0:
		return 'gmx_mpi'
	raise SystemExit('Not found gmx or gmx_mpi.')


GMX = find_gmx()

MDPFILES = sys.path[0] + '/utils'

def get_gaff(ligandfile, molname, ff):
	"""Build a ligand topology and coordinate file from a ligand file using acpype for gaff"""

	paras = {
		'ligandfile': ligandfile,
		'molname': molname,
		'ff': ff,
		'net_charge': Chem.GetFormalCharge(MolFromInput(ligandfile))}

	RC = os.system('''acpype -i {ligandfile} -b {molname} -n {net_charge} -a {ff} -f -o gmx >acpype.{molname}.log 2>&1 '''.format(**paras))
	if RC != 0:
		raise SystemExit('\nERROR!\nFailed to run the acpype. see the %s for details.'%os.path.abspath("acpype.{0}.log\n".format(molname)))
	shutil.copy("LIG.acpype/LIG_GMX.gro", f"{molname}.gro")

def get_cgenff(molecule, molname):
	"""Build a ligand topology and coordinate file from a ligand file using SILCSBIO for cgenff"""


	input_format = molecule.split('.')[-1].lower()

	if input_format != 'mol2':
		obabel_convert(molecule, 'ligand.mol2', QniqueNames=True)
		molecule = 'ligand.mol2'
	
	with open(molecule, 'r') as file:
		content = file.read()

	pattern = r'@<TRIPOS>UNITY_ATOM_ATTR.*?(?=@<TRIPOS>|$)'
	updated_content = re.sub(pattern, '', content, flags=re.DOTALL)

	with open('ligand.mol2', 'w') as file:
		file.write(updated_content) 

	cmd = '''$SILCSBIODIR/cgenff/cgenff_to_gmx.sh mol={} > cgenff.log 2>&1;'''.format(molecule)
	RC = os.system(cmd)
	if RC != 0:
		raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("cgenff.log\n"))
	else:
		shutil.move("posre.itp", f"posre_{molname}.itp")
		shutil.move(f"{Path(molecule).stem}_gmx.top", f"{molname}.top")

		obabel_convert('{}_gmx.pdb'.format(Path(molecule).stem), 'LIG.gro')


def get_openff(molecule, molname):	####https://github.com/openforcefield/openff-toolkit
	"""Build a ligand topology and coordinate file from a ligand file using openff"""


	# Use the OpenFF Toolkit to generate a molecule object from a SMILES pattern
	molecule = Molecule.from_file(molecule)

	# Create an OpenFF Topology object from the molecule
	topology = Topology.from_molecules(molecule)

	# Load the latest OpenFF force field release: version 2.1.0, codename "Sage"
	forcefield = ForceField('openff-2.1.0.offxml')

	# Create an Interchange object
	out = Interchange.from_smirnoff(force_field=forcefield, topology=topology)

	# write to GROMACS files
	out.to_gro("{}.gro".format(molname))
	out.to_top("openff_{}.itp".format(molname))
	#AbstractTopol.writeGromacsTopolFiles(out)
   
########## Get atomtypes from itp files
def get_atomtypes(itp_files, ff):

	'''Combine [ atomtypes ] from **itp_files** a list of itp files '''

	p = False
	atomtypes = []
	for f in itp_files:
		resname = Path(f).stem
		with open(f, 'r') as f:
			for line in f:
				if re.search('\[ atomtypes \]', line):
					p = True
					atomtypes.append(line)
					continue
				elif re.search('\[ moleculetype \]', line):
					p = False
					break
				if p:
					if line.strip() != '':
						if ff == 'openff':
							line_rn = line.replace("MOL0","")
							atomtypes.append(line_rn)
						else:
							atomtypes.append(line)

	return ''.join(map(str, list(dict.fromkeys(atomtypes))))


########## Getting itp file
def mol_itp(itp_file, resname, output_itp, ff):
	p = False
	rename = False
	with open(output_itp, 'w') as output:
		with open(itp_file, 'r') as file:
			for line in file:
				if re.search('\[ moleculetype \]', line):
					output.write(line)
					output.write('''{}              3\n'''.format(resname))
					continue
				elif line[0] == "#":
					p = False
				elif re.search('\[ system \]', line):
					p = False
					break
				elif re.search('\[ atoms \]', line):
					p = True
					rename = True
					output.write(line)
				elif re.search('\[ bonds \]', line) or re.search('\[ pairs \]', line):
					rename = False
					output.write(line)
				elif p == True and rename == False:
					output.write(line)
				elif p == True and rename == True:
					if line[0] != ";" and line[0] != "#" and line[0] != "[" and line.strip() != "":
						line_list = line.split()
						res_name = line_list[3]
						line_rn = line.replace(res_name,resname)
						output.write(line.replace(res_name,resname))


def protein_itp(itp_file, molname, output_itp):
	p = False
	with open(output_itp, 'w') as output:
		with open(itp_file, 'r') as file:
			for line in file:
				if re.search('\[ moleculetype \]', line):
					output.write(line)
					output.write(''';name            nrexcl 
{}              3\n'''.format(molname))
					continue
				elif re.search('\[ atoms \]', line):
					p = True
					output.write(line)
				elif line[0] == "#":
					p = False
				elif re.search('\[ system \]', line):
					p = False
					break
				elif p == True:
					output.write(line)


def get_topol(pro_itp, lig_itp,  ff='gaff', protein_FF='amber99sb-ildn'):

	try:
	    os.makedirs('md_setup')
	    print("Directory 'md_setup' created successfully.")
	except FileExistsError:
	    print("\nWarning: Directory 'md_setup' already exists !\n")


	if ff in {'gaff', 'gaff2', 'openff'}:
		initial = """
; Include forcefield parameters
#include "{}.ff/forcefield.itp"\n
	""".format(protein_FF)
		water = '{}.ff/tip3p.itp'.format(protein_FF)
	elif ff == 'cgenff':
		initial = """
; Include forcefield parameters
#include "./charmm36.ff/forcefield.itp"
;
; Include ligand specific parameters
# include "./charmm36.ff/lig_ffbonded.itp"
	"""
		water = './charmm36.ff/tip3p.itp'
		shutil.move('charmm36.ff', "md_setup/charmm36.ff")
	else:
		raise ValueError("Invalid value for -FF")

	waternumber = ''
	with open(pro_itp, 'r') as file:
		last_line = file.readlines()[-1]
		elements = last_line.split()
		if last_line.startswith('SOL'):
			waternumber = f'SOL {elements[1]}'

	template = """{0}
{1}

#include "protein.itp"

#include "LIG.itp"

#include "{2}"\n

[ system ]
Protein-Ligand\n
[ molecules ]
protein 1
LIG 1
{3}
""".format(initial, get_atomtypes([lig_itp], ff), water, waternumber)

	f = open("md_setup/topol.top", "w")
	f.write(template)
	f.close()

	mol_itp(lig_itp, 'LIG', "md_setup/LIG.itp", ff)
	protein_itp(pro_itp, 'protein', "md_setup/protein.itp")
	shutil.move('protein.gro', "md_setup/protein.gro")

	try:
	    if ff in {'gaff', 'gaff2'}:
	    	shutil.copy('LIG.acpype/LIG_GMX.gro', "md_setup/LIG.gro")
	    else:
	    	shutil.copy('LIG.gro', "md_setup/LIG.gro")
	except FileExistsError:
	    print("Directory 'md_setup' can not be created. The directory already exists")    
	
	combine_gro_files("md_setup/protein.gro", "md_setup/LIG.gro", "md_setup/complex.gro")
	
def combine_gro_files(file1_path, file2_path, output_path, box='10 10 10'):
	with open(file1_path) as file1:
		file1_lines = file1.readlines()
	with open(file2_path) as file2:
		file2_lines = file2.readlines()

	num_atoms1 = int(file1_lines[1])
	num_atoms2 = int(file2_lines[1])
	total_atoms = num_atoms1 + num_atoms2

	# Create the header for the combined file
	title = "Combined Gro File"
	header = f"{title}\n{total_atoms}\n"

	# Combine the atom information from both files
	atoms = []
	for i in range(2, num_atoms1 + 2):
		if 'HOH' not in file1_lines[i]:
			atoms.append(file1_lines[i])
	for i in range(2, num_atoms2 + 2):
		atoms.append(file2_lines[i][:5] + "LIG".ljust(5) + file2_lines[i][10:] )
	for i in range(2, num_atoms1 + 2):
		if 'HOH' in file1_lines[i]:
			atoms.append(file1_lines[i])
	# Write the combined file
	with open(output_path, "w") as output_file:
		output_file.write(header)
		output_file.writelines(atoms)
		output_file.write(box)

def replace_GROcoor(gro_file1, gro_file2, output_path):
	with open(gro_file1) as file1:
		file1_lines = file1.readlines()
	with open(gro_file2) as file2:
		file2_lines = file2.readlines()

	num_atoms = int(file1_lines[1])

	# Create the header for the combined file
	title = "LIG Gro File"
	header = f"{title}\n{num_atoms}\n"
	box = file2_lines[-1]

	with open(output_path, "w") as output_file:
		output_file.write(header)
		for i in range(2, num_atoms + 2):
				line1 = file1_lines[i]
				line2 = file2_lines[i]
				new_line = "{:5d}{:<5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(1, 'LIG', str(line2[10:15]).strip(), i-1, float(str(line1[20:27]).strip()), float(str(line1[28:35]).strip()), float(str(line1[36:43]).strip()))
				output_file.write(new_line)
		output_file.write(box)


def gmx_pdb2gmx(pdbfile, outcoord='protein.gro', outtop='topol.top', protein_FF='amber99sb-ildn', water='tip3p', ignh=True):
	"""Build a protein topology and coordinate file from a PDB file"""
	paras = {
		'gmx':GMX,
		'pdbfile':pdbfile,
		'outfile': outcoord,
		'topolfile': outtop,
		'forcefield': protein_FF,
		'water': water,
		'ignh':ignh}
	cmd = '{gmx} pdb2gmx -f {pdbfile} -o {outfile} -p {topolfile} -ff {forcefield} -water {water} -ignh {ignh} > gromacs.log 2>&1'.format(**paras)
	RC = os.system(cmd)
	if RC != 0:
		raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("gromacs.log\n"))

def gmx_trjcat(idx, time, outtrj):
	"""Build a protein topology and coordinate file from a PDB file"""

	cmd = 'echo Protein System|{0} trjconv -s md_setup/complex.gro -f {1}.trr -o {1}.xtc -pbc nojump -ur compact -center > gromacs1.log 2>&1'.format(GMX, idx)
	RC = os.system(cmd)
	if RC != 0:
		raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("gromacs.log\n"))

	cmd = 'echo Protein System|{0} trjconv -s md_setup/complex.gro -f {1}.xtc -o {1}x.xtc -fit rot+trans > gromacs2.log 2>&1'.format(GMX, idx)
	RC = os.system(cmd)
	if RC != 0:
		raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("gromacs.log\n"))
		
	cmd = 'echo {3}|{0} trjcat -f {1}x.xtc -o {2}.xtc -settime 1 > gromacs.log 2>&1'.format(GMX, idx, outtrj, time)
	RC = os.system(cmd)
	if RC != 0:
		raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("gromacs.log\n"))
		
def gmx_mdrun(idx, md_setup):
	"""gmx grompp (the gromacs preprocessor) reads a molecular topology file, checks the validity of the file, expands the topology from a molecular description to an atomic description."""
	cmd = '{0} grompp -f {1} -o {2}.tpr -c {2}x.gro -p {3}/topol.top -maxwarn 1 > gromacs.log 2>&1'.format(GMX, MDPFILES + '/em.mdp', idx, md_setup)
	RC = os.system(cmd)
	if RC != 0:
		raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("gromacs.log\n"))

	"""Performs Energy Minimization"""
	cmd = '{} mdrun -deffnm {} -nt 1 > gromacs.log 2>&1'.format(GMX, idx)
	RC = os.system(cmd)
	if RC != 0:
		raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("gromacs.log\n"))

def gmx_grompp(md_setup):
	"""gmx grompp (the gromacs preprocessor) reads a molecular topology file, checks the validity of the file, expands the topology from a molecular description to an atomic description."""
	cmd = '{0} grompp -f {1} -o {2}/complex.tpr -c {2}/complex.gro -p {2}/topol.top -maxwarn 1 > gromacs.log 2>&1'.format(GMX, MDPFILES + '/em.mdp', md_setup)
	RC = os.system(cmd)
	if RC != 0:
		raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("gromacs.log\n"))
		
def mda_edr(edrfile, term):

	aux = mda.auxiliary.EDR.EDRReader(edrfile)
	
	#Using only last 100 steps and converting to kcal/mol
	dat = pd.DataFrame.from_dict(aux.get_data(term)).tail(100).mean()*0.2390057361
	return float(dat[term])
			
def system_setup(receptor, ligand, proteinFF='amber99sb-ildn', FF='gaff'):
	if FF == 'openff':
		gmx_pdb2gmx(receptor, outcoord='protein.gro', outtop='protein.top', protein_FF=proteinFF, ignh=False)
		get_openff(ligand, 'LIG')
		get_topol('protein.top', 'openff_LIG.itp', ff=FF, protein_FF=proteinFF)


	if FF in {'gaff', 'gaff2'}:
		gmx_pdb2gmx(receptor, outcoord='protein.gro', outtop='protein.top', protein_FF=proteinFF, ignh=False)
		get_gaff(ligand, 'LIG', FF)
		get_topol('protein.top', 'LIG.acpype/LIG_GMX.itp', ff=FF, protein_FF=proteinFF)


	if FF == 'cgenff':
		if 'SILCSBIODIR' not in os.environ or not os.environ['SILCSBIODIR']:
			raise ValueError("SILCSBIODIR environment variable is not set.")
		else:
			get_cgenff(ligand, 'LIG')
			gmx_pdb2gmx(receptor, outcoord='protein.gro', outtop='protein.top', protein_FF='charmm36', ignh=False)
			get_topol('protein.top', 'LIG.top', ff=FF)


def minimize(i, mol, trjdir):
	

	
	md = str(uuid.uuid4())
	writer = Chem.SDWriter(f"{md}.sdf")
	mol = Chem.AddHs(mol, addCoords=True)
	writer.write(mol)
	writer.close()
			
	obabel_convert(f"{md}.sdf", f'{md}.gro')
			
	replace_GROcoor(f'{md}.gro', 'md_setup/LIG.gro', f'{md}.gro')
			
	combine_gro_files('md_setup/protein.gro', f'{md}.gro', f'{md}x.gro')

	gmx_mdrun(md, 'md_setup') 
	gmx_trjcat(md, i, f'.{trjdir}/{md}.xtc')
    		
	energy = '{:0.3f}'.format(mda_edr(f'{md}.edr', 'LJ-SR:Protein-LIG') + mda_edr(f'{md}.edr', 'Coul-SR:Protein-LIG'))
			
	data = pd.DataFrame({'Index': [f'{i}'], 'Score': [f"{energy}"]})
	data.to_csv('Scores.csv', mode='a', index=False, header=False)
			
	## Deleting temperory trajectory files
	for f in glob.glob(f"*{md}*"):
		os.remove(f)
