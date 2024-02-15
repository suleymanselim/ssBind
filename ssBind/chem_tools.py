#!/usr/bin/python
import os, math
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import Get3DDistanceMatrix
from rdkit.Chem import rdFMCS, rdMolAlign
from rdkit.Chem import rdMolTransforms
from typing import Optional, List
from rdkit.Chem.rdmolfiles import * 
from rdkit.Chem import AllChem
from copy import deepcopy
import MDAnalysis as mda
import numpy as np

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)

def is_file(fname):
	if os.path.isfile(fname):
		return True
	else:
		return False


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ.get("PATH", "").split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
    
    
FILE_PARSERS = {
    "mol": MolFromMolFile,
    "mol2": MolFromMol2File,
    "pdb": MolFromPDBFile,
    "sdf": MolFromMolFile,
}

def MolFromInput(mol_input): 
    #Reading any file format
    if is_file(mol_input) == True:
        content_reader = FILE_PARSERS
        mol_format = os.path.splitext(mol_input)[1][1:]
    else:
    	raise SystemExit(f'\nERROR! Input file ({mol_input}) is not found!!!')
    if mol_format:
        try:
            reader = content_reader[mol_format.lower()]
        except KeyError:
            raise TypeError(
                f"Molecule format {mol_format} not supported. "
                f"Supported formats: " + ", ".join(FILE_PARSERS))
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

def replace_coor(input_sdf, input_mol2, output): 
    #Replacing coodinates from sdf to mol2 file.
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

    
def obabel_convert(input_file, output_file, resname=None, ph=None, QniqueNames=False): 
    #Replacing coodinates from sdf to mol2 file.
    from openbabel import openbabel
    
    openbabel.obErrorLog.StopLogging()

    mol = openbabel.OBConversion()
    input_format = input_file.split('.')[-1].lower()
    output_format = output_file.split('.')[-1].lower()
    
    mol.SetInAndOutFormats(input_format, output_format)
    mol_mol = openbabel.OBMol()
    
    
    mol.ReadFile(mol_mol, input_file) 

    if resname != None:
    	residue = next(openbabel.OBResidueIter(mol_mol))
    	residue.SetName(resname)
    	mol_mol.SetTitle(resname)
    
    if ph != None:
    	mol_mol.AddHydrogens(polaronly=False, correctForPh=True, pH=7.4)
    
    for bond in openbabel.OBMolBondIter(mol_mol):
      bond.SetAromatic(False)

    for atom in openbabel.OBMolAtomIter(mol_mol):
      atom.SetAromatic(False)
          
    mol.WriteFile(mol_mol, output_file)

    if QniqueNames:
        makeQniqueNames(output_file)


def makeQniqueNames(mol2_file):
	number = {}
	with open(mol2_file, 'r') as x:
		lines = x.readlines()

	p = False
	with open(mol2_file, 'w') as f:
		for line in lines:
			if line.startswith('@<TRIPOS>ATOM'):
				f.write(line)
				p = True
			elif line.startswith('@<TRIPOS>') and p:
				f.write(line)
				p = False
			elif p:
				atom_data = line.split()
				letter = atom_data[1]
				if letter not in number:
					number[atom_data[1]] = 1
				else:
					number[atom_data[1]] += 1
				atom_data[1] = f"{atom_data[1]}{number[atom_data[1]]}"
				atom_data[7] = 'LIG'
				formatted_line = '{0:6d} {1:<8s} {2:10.4f} {3:10.4f} {4:10.4f} {5:<8s} {6:6d} {7:<8s} {8:10.6f}\n'.format(\
                    int(atom_data[0]), atom_data[1], float(atom_data[2]), float(atom_data[3]), float(atom_data[4]),\
                    atom_data[5], int(atom_data[6]), atom_data[7], float(atom_data[8]))
				f.write(formatted_line)
			else:
				f.write(line)


def getDihedralMatches(mol): 
    #Getting uniq dihedral matches.
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

def MCS_AtomMap(query, ref): 
    #Building atom map for the MCS.
    mcs = rdFMCS.FindMCS([query, ref], completeRingsOnly=True, matchValences=True)
    submol = Chem.MolFromSmarts(mcs.smartsString)
    refMatch = ref.GetSubstructMatch(submol)
    queryMatch = query.GetSubstructMatch(submol)
    Amap = []
    for i in range(len(refMatch)):
        Amap.append((queryMatch[i], refMatch[i]))
    return Amap
    
def get_uniqueDihedrals(refmol, mol): 
    #Getting uniq dihedral matches not found in MCS using atom index

    DM_mol = getDihedralMatches(mol)
    submol = Chem.MolFromSmarts(
            rdFMCS.FindMCS([mol, refmol], completeRingsOnly=True, matchValences=True).smartsString
            )
    queryMatch = mol.GetSubstructMatch(submol)
    uniqueDihedrals = []
    for a in DM_mol:
        if a[1] not in queryMatch:
            uniqueDihedrals.append((a))
        elif a[2] not in queryMatch:
            uniqueDihedrals.append((a))
        else:
            continue
    return uniqueDihedrals
    
def degreeRange(inc): 
    #Producing angle degrees for dihedral sampling.
    degrees = []
    deg = 0
    while deg < 360.0:
        rad = math.pi*deg / 180.0
        degrees.append(rad)
        deg += inc
    return degrees
    #The code was adapted from David Koes https://github.com/dkoes/rdkit-scripts/blob/master/rdallconf.py

def gen_conf_angle(j, mol_Dihedrals, mol, reflig):
    # Generate conformers using angle
    intD = 0
    for i in mol_Dihedrals:
        rdMolTransforms.SetDihedralRad(mol.GetConformer(),*i,value=j[intD])
        intD += 1
    AlignMol(mol, reflig, atomMap=MCS_AtomMap(mol, reflig))
    outf = open('conformers.sdf','a')
    sdwriter = Chem.SDWriter(outf)
    sdwriter.write(mol)
    sdwriter.close()
    outf.close()

def gen_conf_rdkit(mol: Chem.rdchem.Mol, ref_mol: Chem.rdchem.Mol, j):
    # Generate conformers using rdkit with constrained embed
    ref_smi = Chem.MolToSmiles(
        Chem.MolFromSmarts(
            rdFMCS.FindMCS([mol, ref_mol], completeRingsOnly=True, matchValences=True).smartsString
        )
    )
    # Creating core of reference ligand 
    core_with_wildcards = AllChem.ReplaceSidechains(ref_mol, Chem.MolFromSmiles(ref_smi))
    core1 = AllChem.DeleteSubstructs(core_with_wildcards, Chem.MolFromSmiles('*'))
    core1.UpdatePropertyCache()

    mol.RemoveAllConformers()
    outmol = deepcopy(mol)
    mol_wh = Chem.AddHs(mol)

    outf = open('conformers.sdf','a')
    temp_mol = Chem.Mol(mol_wh)  
    AllChem.ConstrainedEmbed(temp_mol, core1, randomseed=j)
    temp_mol = Chem.RemoveHs(temp_mol)
    conf_idx = outmol.AddConformer(temp_mol.GetConformer(0), assignId=True)
    sdwriter = Chem.SDWriter(outf)
    sdwriter.write(outmol, conf_idx)
    sdwriter.close()
    outf.close()


def CheckRMS(sdfmol, ref, rms=0.2): 
    ##Filtering identical conformations
    try:
        outf = Chem.SDMolSupplier(sdfmol)
        for i, mol in enumerate(outf):
            if GetBestRMS(outf[i], ref) < rms:
                return True
            else:
                continue
        return False
    except OSError:
        return False 

def steric_clash(mol):
    from rdkit.Chem.Draw import rdMolDraw2D
    ##Identify stearic clashes based on mean bond length
    ditancelist = Get3DDistanceMatrix(mol)[0]
    for i in range(1, len(ditancelist)):
        if ditancelist[i] < 0.5 * rdMolDraw2D.MeanBondLength(mol):
            return True
        else:
            continue
    return False

def distance(receptor, ligand, cutoff=1.5): 
    ##Calculating min distance between protein and ligand.
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances

    protein = mda.Universe(receptor)
    ligand = mda.Universe(ligand)

    atom1 = protein.select_atoms("not name H*")
    atom2 = ligand.select_atoms("not name H*")

    distances = mda.analysis.distances.distance_array(atom1.positions, atom2.positions)

    if distances.min() < cutoff:
        return True
    else:
        return False

def filtering(mol, receptor, cutoff=1.5, rms=0.2):
    outf = open('filtered.sdf','a')
    if steric_clash(mol):
        return
    elif distance(receptor, mol, cutoff):
        return
    elif CheckRMS('filtered.sdf', mol, rms):
        return
    else:
        sdwriter = Chem.SDWriter(outf)
        sdwriter.write(mol)
        sdwriter.close()
        outf.close()

def clustering_poses(inputfile, ref, csv_scores, output, binsize, distThresh, numbin):
    import MDAnalysis as mda
    from gmx_tools import gmx_grompp
    from rdkit.Chem.Descriptors3D import Asphericity

    from MDAnalysis.analysis import pca, align
    import matplotlib.pyplot as plt
    import seaborn as sns
    from rpy2 import robjects
    import rpy2.robjects.lib.ggplot2 as ggplot2

    input_format = inputfile.split('.')[-1].lower()

    if input_format != 'sdf':
        
        gmx_grompp('md_setup')
        
        u = mda.Universe('md_setup/complex.tpr', 'trjout.xtc') 	
        elements = mda.topology.guessers.guess_types(u.atoms.names)
        u.add_TopologyAttr('elements', elements)
        atoms = u.select_atoms('resname LIG')
        
        #atoms.write("output-traj.pdb", frames='all', multiframe=True, bonds='conect')
        
        sdwriter = Chem.SDWriter('traj-output.sdf')
        for ts in u.trajectory:
            sdwriter.write(atoms.convert_to("RDKIT"))
        
        sdwriter.close()
        confs = Chem.SDMolSupplier('traj-output.sdf')


    else:
        confs = Chem.SDMolSupplier(inputfile, sanitize=False)
        u = mda.Universe(confs[0], confs)     

    pc = pca.PCA(u, select='not (name H*)',
                 align=False, mean=None,
                 n_components=None).run()

    atoms = u.select_atoms('not (name H*)')

    transformed = pc.transform(atoms, n_components=3)
    transformed.shape

    df = pd.DataFrame(transformed,
                      columns=['PC{}'.format(i+1) for i in range(3)])


    df['Index'] = df.index * u.trajectory.dt

    scoredata = pd.read_csv(csv_scores, delimiter=',', header=0)
    PCA_Scores = pd.merge(df, scoredata, on='Index')
    PCA_Scores.to_csv('PCA_Scores.csv', index=False, header=True)

    pcs = ['PC1', 'PC2', 'PC3']
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):

            index_data = []

            robjects.r(f'''
            library(ggplot2)

            df <- read.table('PCA_Scores.csv', header=TRUE, sep=",")

            p <- ggplot(df, aes(x={pcs[i]}, y={pcs[j]}, z=Score)) +
                stat_summary_2d(fun=mean, binwidth = {binsize}) 

            plot_data <- ggplot_build(p)
            bin_data <- plot_data$data[[1]]
            write.csv(bin_data, file = '{pcs[i]}_{pcs[j]}.csv', row.names = FALSE)
            ''')

            data = pd.read_csv(f'{pcs[i]}_{pcs[j]}.csv', delimiter=',', header=0)
            raw_data = pd.read_csv('PCA_Scores.csv', delimiter=',', header=0)

            df_sorted = data.sort_values(by='value')
            top_bins = df_sorted.head(numbin)
            extracted_data = top_bins[['xmin', 'xmax','ymin', 'ymax']]

            for a, rowa in extracted_data.iterrows():
                for b, rowb in raw_data.iterrows():
                    if rowa['xmin'] < rowb[pcs[i]] < rowa['xmax'] and rowa['ymin'] < rowb[pcs[j]] < rowa['ymax']:
                        index_data.append(rowb)
            os.remove(f'{pcs[i]}_{pcs[j]}.csv')

            robjects.r(f'''
            library(ggplot2)

            mydata <- read.table('PCA_Scores.csv', header=TRUE, sep=",")

            p <- ggplot(mydata, aes(x={pcs[i]}, y={pcs[j]})) +
                stat_bin_2d(binwidth ={binsize}, aes(fill = after_stat(density)))

            plot_data <- ggplot_build(p)
            bin_data <- plot_data$data[[1]]
            write.csv(bin_data, file = '{pcs[i]}_{pcs[j]}.csv', row.names = FALSE)
            ''')

            data = pd.read_csv(f'{pcs[i]}_{pcs[j]}.csv', delimiter=',', header=0)
            raw_data = pd.read_csv('PCA_Scores.csv', delimiter=',', header=0)

            df_sorted = data.sort_values(by='density')
            top_bins = df_sorted.tail(numbin)
            extracted_data = top_bins[['xmin', 'xmax','ymin', 'ymax']]

            for a, rowa in extracted_data.iterrows():
                for b, rowb in raw_data.iterrows():
                    if rowa['xmin'] < rowb[pcs[i]] < rowa['xmax'] and rowa['ymin'] < rowb[pcs[j]] < rowa['ymax']:
                        if any(row['Index'] == rowb['Index'] for row in index_data):
                           index_data.append(rowb)
            os.remove(f'{pcs[i]}_{pcs[j]}.csv')



    cids = []
    index_dict = []
    

    for i, entry in enumerate(index_data):
        cids.append(confs[int(entry['Index'])])
        index_dict.append({i: int(entry['Index'])})

    dists = []
    for i in range(len(cids)):
        for j in range(i):
            rms = rdMolAlign.GetBestRMS(cids[i],cids[j])
            dists.append(rms)

    from rdkit.ML.Cluster import Butina
    clusts = Butina.ClusterData(dists, len(cids), distThresh, isDistData=True, reordering=True)
    
    #from sklearn.cluster import DBSCAN
    #from scipy.spatial.distance import squareform
    #dbscan = DBSCAN(metric='precomputed', eps=0.75, min_samples=5, algorithm = 'auto', n_jobs = 8 )
    #clustering = dbscan.fit(squareform(dists))

    
    PC1 = []
    PC2 = []
    PC3 = []
    
    sdwriter = Chem.SDWriter('clusts.sdf')
    for i in clusts:
        a = index_data[i[0]]
        PC1.append(a['PC1'])
        PC2.append(a['PC2'])
        PC3.append(a['PC3'])
        dict_i = index_dict[i[0]]
        sdwriter.write(confs[int(next(iter(dict_i.values())))])
    sdwriter.close()


    pcx = [PC1, PC2, PC3]
    pcs = ['PC1', 'PC2', 'PC3']
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):

            robjects.r(f'''
            library(ggplot2)
            library(ggdensity)

            df <- read.table('PCA_Scores.csv', header=TRUE, sep=",")
            
            p <- ggplot(df, aes(x={pcs[i]}, y={pcs[j]}, z=Score)) +
                stat_summary_2d(fun=mean, binwidth = {binsize}) +
                scale_fill_gradientn(colours = rainbow(5), limits = c(min(df$Score),max(df$Score))) +
                geom_hdr_lines() +
                labs(fill = "Score") +
                theme(
            panel.border = element_rect(color = "black", fill = 'NA', linewidth = 1.5),
            panel.background = element_rect(fill = NA),
            panel.grid.major = element_line(linewidth = 0),
            panel.grid.minor = element_line(linewidth = 0),
            axis.text = element_text(size = 16, color = "black", face = "bold"),
            axis.ticks = element_line(color = "black", linewidth = 1),
            axis.ticks.length = unit(5, "pt"),
            axis.title.x = element_text(vjust = 1, size = 20, face = "bold"),
            axis.title.y = element_text(angle = 90, vjust = 1, size = 20, face = "bold"),
            legend.text = element_text(size = 10, face="bold"),
            legend.title = element_text(size = 16, hjust = 0, face="bold")
            ) +
            annotate("text", x = c(''' + str(', '.join(map(str, pcx[i]))) + '''), y = c(''' + str(', '.join(map(str, pcx[j]))) + '''), size = 6, 
            fontface = "bold", label = c(''' + str(', '.join(map(str, list(range(1, len(clusts) + 1))))) + f''')) +
            coord_fixed((max(df${pcs[i]})-min(df${pcs[i]}))/(max(df${pcs[j]})-min(df${pcs[j]})))

            ggsave("{pcs[i]}-{pcs[j]}_{output}", width = 7, height = 7)

            ''')

	
def find_nearest_conf_to_average(input_file):
    from rdkit.Geometry import Point3D
    
    # Load the conformations
    u = mda.Universe(input_file[0], input_file)

    # Calculate the average positions
    avg_coordinates = np.mean([u.trajectory[i].positions for i in range(len(u.trajectory))], axis=0)

    mol = input_file[0]
    avg_conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = avg_coordinates[i]
        avg_conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    
    mol.AddConformer(avg_conf, assignId=True)

    # Initialize minimum distance and conf index
    distance = float('inf')
    frame = -1

    # Iterate over each conf to find the closest to the average conf
    for i, mol_i in enumerate(input_file):
        rmsd = rdMolAlign.GetBestRMS(mol, mol_i)
        if rmsd < distance:
            distance = rmsd
            frame = i

    return frame

