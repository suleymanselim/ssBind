#!/usr/bin/python
import os, math
import argparse
import itertools
from rdkit import Chem
from rdkit.Chem.rdMolAlign import AlignMol, GetBestRMS
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolTransforms
from typing import Optional
from rdkit.Chem.rdmolfiles import * 
import multiprocessing as mp
from joblib import Parallel, delayed



#	usage: conformer_generator.py [-h] --ref REF --mol MOL --rec REC
#	                              [--degree DEGREE] [--cutoff CUTOFF_DIST]
#	                              [--rms RMS] [--output OUTPUT]
#
#	options:
#	  -h, --help            show this help message and exit
#	  --ref REF             Referance molecule
#	  --mol MOL             Ligand molecule
#	  --rec REC             PDB file for receptor protein
#	  --degree DEGREE       Amount, in degrees, to enumerate torsions by (default 15.0)
#	  --cutoff CUTOFF_DIST  Cutoff for eliminating any conformer close to protein within cutoff by (default 1.5 A)
#	  --rms RMS             Only keep structures with RMS > CUTOFF (default 1.0 A)
#	  --output OUTPUT       Output sdf file for conformations


RC = os.system('export AMBERHOME=~/miniconda/amber.sh')
if RC != 0:
	raise SystemExit('\nERROR!\nFailed to get AMBERHOME. ')
        
def ParserOptions():
    parser = argparse.ArgumentParser()

    """Parse command line arguments."""
    parser.add_argument("--ref", dest="ref", help="Referance molecule", required=True)   
    parser.add_argument("--mol", dest="mol", help="Ligand molecule", required=True)
    parser.add_argument("--rec", dest="rec", help="PDB file for receptor protein", required=True)
    parser.add_argument("--degree", dest="degree", type=float,help="Amount, in degrees, to enumerate torsions by (default 15.0)", default=15.0) 
    parser.add_argument("--cutoff", dest="cutoff_dist", type=float,help="Cutoff for eliminating any conformer close to protein within cutoff by (default 1.0 A)", default=1.5) 
    parser.add_argument("--rms", dest="rms", type=float,help="Only keep structures with RMS > CUTOFF (default 1.0 A)", default=1.0) 
    parser.add_argument("--output", dest="output", help="Output sdf file for conformations", default='output.sdf')
    parser.add_argument("--cpu", dest="cpu", type=int, help="Number of CPU. If not set, it uses all available CPUs.") 
    args = parser.parse_args()
    return args


FILE_PARSERS = {
    "mol": MolFromMolFile,
    "mol2": MolFromMol2File,
    "pdb": MolFromPDBFile,
    "sdf": MolFromMolFile,
}

def MolFromInput(mol_input): #Reading any file format
    if os.path.isfile(mol_input):
        content_reader = FILE_PARSERS
        mol_format = os.path.splitext(mol_input)[1][1:]
    if mol_format:
        try:
            reader = content_reader[mol_format.lower()]
        except KeyError:
            raise TypeError(
                f"Molecule format {mol_format} not supported. "
                f"Supported formats: " + ", ".join(FILE_PARSERS)
            )
        return reader(mol_input)

    for reader in content_reader.values():
        try:
            mol = reader(mol_input)
        except RuntimeError:
            pass
        else:
            if mol is not None:
                return mol

    raise TypeError(
        f"Could not create an RDKit molecule from {mol_input}. "
        "Try passing in a `mol_format`. "
        f"Supported formats: " + ", ".join(FILE_PARSERS)
    ) #The code was adapted from https://pypi.org/project/rdkit-utilities/

def replace_coor(input_sdf, input_mol2, output): #Replacing coodinates from sdf to mol2 file.
    from openbabel import openbabel, OBConversion, vector3
    import random
    
    openbabel.obErrorLog.StopLogging()

    # Read in SDF file and extract coordinates
    sdf = OBConversion()
    sdf.SetInFormat("sdf")
    mol_sdf = openbabel.OBMol()
    sdf.ReadFile(mol_sdf, input_sdf) 

    # Read in MOL2 file and extract coordinates
    mol2 = OBConversion()
    mol2.SetInAndOutFormats("mol2", "mol2")
    mol_mol2 = openbabel.OBMol()
    mol2.ReadFile(mol_mol2, input_mol2) 


    for atom_sdf, atom_mol2 in zip(openbabel.OBMolAtomIter(mol_sdf), openbabel.OBMolAtomIter(mol_mol2)):
      x = float(atom_sdf.x())
      y = float(atom_sdf.y())
      z = float(atom_sdf.z())

      coords = vector3(x, y, z)
      atom_mol2.SetVector(coords)
    
    residue = next(openbabel.OBResidueIter(mol_mol2))
    residue.SetName("LIG")
    mol_mol2.SetTitle("LIG")
    
    mol2.WriteFile(mol_mol2, output)

def get_atom_names(mol, match): #Getting  Tripos atom name for dihedral match. Depricated.
    atom_names = []
    for idx in match:
        atom_data = mol.GetAtomWithIdx(idx).GetPropsAsDict()
        if '_TriposAtomName' in atom_data:
            atom_names.append(atom_data['_TriposAtomName'])
    return atom_names

def getDihedralMatches(mol): #Getting uniq dihedral matches.
    '''return list of atom indices of dihedrals'''
    #this is rdkit's "strict" pattern
    pattern = r"*~[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]~*"
    qmol = Chem.MolFromSmarts(pattern)
    matches = mol.GetSubstructMatches(qmol);
    #these are all sets of 4 atoms, uniquify by middle two
    uniqmatches = []
    seen = set()
    for (a,b,c,d) in matches:
        if (b,c) not in seen:
            seen.add((b,c))
            uniqmatches.append((a,b,c,d))
    return uniqmatches
    #The code was adapted from David Koes https://github.com/dkoes/rdkit-scripts/blob/master/rdallconf.py

def MCS_AtomMap(query, ref): #Building atom map for the MCS.
    mcs = rdFMCS.FindMCS([query, ref])
    submol = Chem.MolFromSmarts(mcs.smartsString)
    refMatch = ref.GetSubstructMatch(submol)
    queryMatch = query.GetSubstructMatch(submol)
    Amap = []
    for i in range(len(refMatch)):
        Amap.append((queryMatch[i], refMatch[i]))
    return Amap
    
def uniqueDihedrals(refmol, mol): #Getting uniq dihedral matches not found in MCS using tripos atom names Depricated.

    DM_refmol = getDihedralMatches(refmol)
    ref_dict = {}

    for match in DM_refmol:
        ref_atom_names = get_atom_names(refmol, match)
        ref_dict[tuple(ref_atom_names)]= match

    DM_mol = getDihedralMatches(mol)
    mol_dict = {}

    for match in DM_mol:
        mol_atom_names = get_atom_names(mol, match)
        mol_dict[tuple(mol_atom_names)]= match

    # Find unique dihedrals in mol
    unique_DA = set(mol_dict.keys()) - set(ref_dict.keys())
    unique_list = list(unique_DA)
    
    Dihedrals = []
    for ul in unique_list:
        Dihedrals.append(mol_dict[ul])
    return Dihedrals

def get_uniqueDihedrals(refmol, mol): #Getting uniq dihedral matches not found in MCS suing atom index

    DM_mol = getDihedralMatches(mol)
    mcs = rdFMCS.FindMCS([mol, refmol])
    submol = Chem.MolFromSmarts(mcs.smartsString)
    queryMatch = mol.GetSubstructMatch(submol)
    
    uniqueDihedrals = []
    for (a,b,c,d) in DM_mol:
        if (b or c) not in queryMatch:
            uniqueDihedrals.append((a,b,c,d))
    return uniqueDihedrals
    
def degreeRange(inc): #Producing angle degrees for dihedral sampling.
    degrees = []
    deg = 0
    while deg < 360.0:
        rad = math.pi*deg / 180.0
        degrees.append(rad)
        deg += inc
    return degrees
    #The code was adapted from David Koes https://github.com/dkoes/rdkit-scripts/blob/master/rdallconf.py

def distance(receptor, ligand): #Calculating min distance between protein and ligand.
	import MDAnalysis as mda
	from MDAnalysis.analysis import distances
	
	protein = mda.Universe(receptor)
	ligand = mda.Universe(ligand)
	
	atom1 = protein.select_atoms("not name H*")
	atom2 = ligand.select_atoms("not name H*")
	
	distances = mda.analysis.distances.distance_array(atom1.positions, atom2.positions)
	return distances.min()
	
def run_acpype(sdf_filename): #Running Acpype to get GAFF parameters and charges.
    sdf = Chem.SDMolSupplier(sdf_filename)
    writer = Chem.SDWriter("ligand.sdf")
    mol = Chem.AddHs(sdf[0], addCoords=True)
    formal_charge = Chem.GetFormalCharge(mol)
    writer.write(mol)
    writer.close()
    
    RC = os.system('''acpype -i ligand.sdf -b LIG -n {} -f -o gmx >acpype.log 2>&1 '''.format(formal_charge))
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the acpype. See the %s for details.'%os.path.abspath("acpype.log\n"))
    return

def run_tleap(FF='ff14SB', gaff='gaff2', water='tip3p', mbondi='mbondi2', receptor='receptor.pdb', ligand='lig.bcc.mol2', frcmod='LIG.acpype/LIG_AC.frcmod'):
    with open('tleap.in', 'w') as f: 
        f.writelines('source leaprc.protein.{}\n'.format(FF))
        f.writelines('source leaprc.{}\n'.format(gaff))
        f.writelines('source leaprc.water.{}\n'.format(water))
        f.writelines('set default PBradii {}\n'.format(mbondi))
        f.writelines('PROT=loadpdb {}\n'.format(receptor))
        f.writelines('saveamberparm PROT receptor.prmtop receptor.inpcrd\n')
        f.writelines('LIG=loadmol2 {}\n'.format(ligand))
        f.writelines('loadamberparams {}\n'.format(frcmod))
        f.writelines('saveamberparm LIG ligand.prmtop ligand.inpcrd\n')
        f.writelines('complex = combine {PROT LIG}\n')
        f.writelines('saveamberparm complex complex.prmtop complex.inpcrd\n')
        f.writelines('quit\n')
    
    RC = os.system('''tleap -f tleap.in  > tleap.log 2>&1 ''')
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the tleap. See the {} for details.'.format(os.path.abspath("tleap.log\n")))
    return

def run_minimization():
    with open('min.in', 'w') as f: 
        f.writelines('Initial minimisation of rec-lig complex\n')
        f.writelines('&cntrl\n')
        f.writelines('imin=1, maxcyc=2500, ncyc=100,\n')
        f.writelines('cut=16, ntb=0, igb=0,\n')
        f.writelines('ntpr=500, ntwx=500, ntwr=500, drms=0.01\n')
        f.writelines('''ibelly=1, bellymask=':LIG <@10'\n''')
        f.writelines('&end\n')
        f.writelines('/\n')
    
    RC = os.system('''sander -O -i min.in -p complex.prmtop -c complex.inpcrd -r min.rst -ref complex.inpcrd -o minim.out > minim.log 2>&1 ''')
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the minimization. See the {} for details.'.format(os.path.abspath("minim.log\n")))
    return
        
def run_gbsa():
    with open('gbsa.in', 'w') as f: 
        f.writelines('mmgbsa  analysis\n')
        f.writelines('&general\n')
        f.writelines('verbose=2, keep_files=0, netcdf=1,\n')
        f.writelines('/\n')
        f.writelines('&gb\n')
        f.writelines('/\n')
    
    RC = os.system('''MMPBSA.py -O -i gbsa.in -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop -y  min.rst > MMPBSA.log 2>&1 ''')
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the MMPBSA.py. See the {} for details.'.format(os.path.abspath("MMPBSA.log\n")))
    return    

def cpptraj_LIGmol2(output_mol2):
    with open('cpptraj.in', 'w') as f: 
        f.writelines('parm complex.prmtop\n')
        f.writelines('trajin min.rst\n')
        f.writelines('strip !(:LIG)\n')
        f.writelines('trajout {} sybyltype\n'.format(output_mol2))
    
    RC = os.system('''cpptraj -i cpptraj.in  > cpptraj.log 2>&1 ''')
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the cpptraj. See the {} for details.'.format(os.path.abspath("cpptraj.log\n")))
    return  
    
def CheckRMS(sdfmol, ref, rms=1.0): #Filtering identical conformations
	if os.path.exists(sdfmol):
		outf = Chem.SDMolSupplier(sdfmol)
		for i, mol in enumerate(outf):
			if GetBestRMS(outf[i], ref) < rms:
				return True
			else:
				continue
		return False
	else:
		return False

def gen_confs(j, mol_Dihedrals, ligand, reflig, receptor, cutoff_dist, output, rms):
	intD = 0
	mol = MolFromInput(ligand)
	for i in mol_Dihedrals:
		rdMolTransforms.SetDihedralRad(mol.GetConformer(),*i,value=j[intD])
		intD += 1
	AlignMol(mol, refmol, atomMap=MCS_AtomMap(mol, refmol))
	min_dist = distance(receptor, mol)
	if  min_dist > cutoff_dist and False == CheckRMS(output, mol, rms):
		outf = open(output,'a')
		sdwriter = Chem.SDWriter(outf)
		sdwriter.write(mol)
		sdwriter.close()
		outf.close()

	
if __name__ == '__main__':

	args = ParserOptions()
	
	if os.path.exists(args.output):
		raise SystemExit('\nWarning!\nThe output file exists. Check the file {}'.format(args.output))
	
	refmol = MolFromInput(args.ref)
	input_file = MolFromInput(args.mol)
	
	molDihedrals = get_uniqueDihedrals(refmol, input_file)
	inputs = itertools.product(degreeRange(args.degree),repeat=len(molDihedrals))

	if(len(molDihedrals) > 3):
		print("\nWarning! Too many torsions ({})".format(len(molDihedrals)))
	print('\nConformational sampling is running for {} dihedrals.'.format(len(molDihedrals)))  

	if args.cpu is not None:
		nprocs = args.cpu
	else:
		nprocs = mp.cpu_count()
	print(f"\nNumber of CPU cores in use for conformer generation: {nprocs}")
    
    
	pool = mp.Pool(processes=nprocs)
	pool.starmap(gen_confs, [(j, molDihedrals, args.mol, refmol, args.rec, args.cutoff_dist, args.output, args.rms) for j in inputs])
	
	
	print('\n{} conformers have been generated.\n'.format(len(Chem.SDMolSupplier(args.output))))
	
	run_acpype(args.output)

	with open('GBSA_results.txt', 'w') as result_output:
				
		confs = Chem.SDMolSupplier(args.output)
		
		for i, mol in enumerate(confs):
			
			# Write conformation to its own SDF file
			outfile = f'molecule_{i}.sdf'
			w = Chem.SDWriter(outfile)
			w.write(confs[i])
			w.close()
			
			replace_coor(outfile, "LIG.acpype/LIG_bcc_gaff2.mol2", 'lig.bcc.mol2')
			run_tleap(receptor=args.rec)
			run_minimization()
			cpptraj_LIGmol2('ligandLIG.mol2')
			LIGmol2 = MolFromInput('ligandLIG.mol2')
			if  False == CheckRMS('minimized.sdf', LIGmol2, args.rms):
				outf = open('minimized.sdf','a')
				sdwriter = Chem.SDWriter(outf)
				sdwriter.write(LIGmol2)
				sdwriter.close()
				outf.close()			
				run_gbsa()
				with open("FINAL_RESULTS_MMPBSA.dat", 'r') as f:
					for line in f:
						if line.startswith('DELTA TOTAL'):
							parts = line.split()
							energy = parts[2]
							result_output.write(f"molecule_{i} {energy}" + "\n")
			os.remove("molecule_{i}.sdf")
