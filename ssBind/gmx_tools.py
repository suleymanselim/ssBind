import os, shutil, sys, re
from pathlib import Path


def find_gmx():
	RC = os.system('gmx -h >/dev/null 2>&1')
	if RC == 0:
		return 'gmx'
	RC = os.system('gmx_mpi -h >/dev/null 2>&1')
	if RC == 0:
		return 'gmx_mpi'
	raise SystemExit('Not found gmx or gmx_mpi.')


GMX = find_gmx()


def get_gaff(ligandfile, molname):
	"""Build a ligand topology and coordinate file from a ligand file using acpype for gaff"""
	from rdkit import Chem
	from chem_tools import MolFromInput
	paras = {
		'ligandfile': ligandfile,
		'molname': molname,
		'net_charge': Chem.GetFormalCharge(MolFromInput(ligandfile))}

	RC = os.system('''acpype -i {ligandfile} -b {molname} -n {net_charge} -f -o gmx >acpype.{molname}.log 2>&1 '''.format(**paras))
	if RC != 0:
		raise SystemExit('\nERROR!\nFailed to run the acpype. see the %s for details.'%os.path.abspath("acpype.{0}.log\n".format(molname)))
	shutil.copy("LIG.acpype/LIG_GMX.gro", f"{molname}.gro")

def get_cgenff(molecule, molname):
	"""Build a ligand topology and coordinate file from a ligand file using SILCSBIO for cgenff"""
	from chem_tools import obabel_convert, makeQniqueNames

	input_format = molecule.split('.')[-1].lower()

	if input_format != 'mol2':
		obabel_convert(molecule, 'ligand.mol2', resname='LIG')
		molecule = 'ligand.mol2'

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
	from acpype.topol import AbstractTopol, ACTopol, MolTopol, header
	from openff.toolkit.topology.molecule import Molecule

	# Use the OpenFF Toolkit to generate a molecule object from a SMILES pattern
	molecule = Molecule.from_file(molecule)

	# Create an OpenFF Topology object from the molecule
	from openff.toolkit.topology import Topology
	topology = Topology.from_molecules(molecule)

	# Load the latest OpenFF force field release: version 2.1.0, codename "Sage"
	from openff.toolkit.typing.engines.smirnoff import ForceField
	forcefield = ForceField('openff-2.1.0.offxml')

	# Create an Interchange object
	from openff.interchange import Interchange
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
							line_rn = line.replace("MOL0","LIG")
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
						if ff == 'openff':
							atom_name = line_list[1]
							output.write(line_rn.replace(atom_name,atom_name + "_{}".format(resname)))
						else:
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
	    print("Directory 'md_setup' can not be created.")


	if ff == 'gaff' or ff == 'openff':
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
# include "./charmm36.ff/LIG_ffbonded.itp"
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

    
	
def combine_gro_files(file1_path, file2_path, output_path, box='1 1 1'):
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
		output_file.write('1 1 1')

def gro_coordinates(line):
    fields = []
    fields.append(line[21:28].strip()) # x coord
    fields.append(line[29:36].strip()) # y coord
    fields.append(line[37:44].strip()) # z coord
    if (all([f != '' for f in fields])): # check for empty fields
        return all([_isint(fields[0]), _isfloat(fields[1]), _isfloat(fields[2]))])
    else:
        return 0


def gmx_pdb2gmx(pdbfile, outcoord='protein.gro', outtop='topol.top', protein_FF='amber99sb-ildn', water='tip3p', ignh=False):
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

def gmx_mdrun():
	"""gmx grompp (the gromacs preprocessor) reads a molecular topology file, checks the validity of the file, expands the topology from a molecular description to an atomic description."""
	cmd = '{} grompp -f em.mdp -o em.tpr -c complex.gro -n index.ndx -p topol.top > gromacs.log 2>&1'.format(GMX)
	RC = os.system(cmd)
	if RC != 0:
		raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("gromacs.log\n"))

	"""Performs Energy Minimization"""
	cmd = '{} mdrun -deffnm em -nt 1 > gromacs.log 2>&1'.format(GMX)
	RC = os.system(cmd)
	if RC != 0:
		raise SystemExit('\nERROR! see the log file for details %s'%os.path.abspath("gromacs.log\n"))
		
def system_setup(receptor, ligand, proteinFF='amber99sb-ildn', FF='gaff'):
	if FF == 'openff':
		gmx_pdb2gmx(receptor, outcoord='protein.gro', outtop='protein.top', protein_FF=proteinFF, ignh=False)
		get_openff(ligand, 'LIG')
		get_topol('protein.top', 'openff_LIG.itp', ff=FF, protein_FF=proteinFF)


	if FF == 'gaff':
		gmx_pdb2gmx(receptor, outcoord='protein.gro', outtop='protein.top', protein_FF=proteinFF, ignh=False)
		#get_gaff(ligand, 'LIG')
		get_topol('protein.top', 'LIG.acpype/LIG_GMX.itp', ff=FF, protein_FF=proteinFF)


	if FF == 'cgenff':
		if 'SILCSBIODIR' not in os.environ or not os.environ['SILCSBIODIR']:
			raise ValueError("SILCSBIODIR environment variable is not set.")
		else:
			get_cgenff(ligand, 'LIG')
			gmx_pdb2gmx(receptor, outcoord='protein.gro', outtop='protein.top', protein_FF='charmm36', ignh=False)
			get_topol('protein.top', 'LIG.top', ff=FF)

	
