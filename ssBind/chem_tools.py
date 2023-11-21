#!/usr/bin/python
import os, math
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolAlign import AlignMol, GetBestRMS
from rdkit.Chem.rdmolops import Get3DDistanceMatrix
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolTransforms
from typing import Optional, List
from rdkit.Chem.rdmolfiles import * 
from joblib import Parallel, delayed
from rdkit.Chem import AllChem
from copy import deepcopy



FILE_PARSERS = {
    "mol": MolFromMolFile,
    "mol2": MolFromMol2File,
    "pdb": MolFromPDBFile,
    "sdf": MolFromMolFile,
}

def MolFromInput(mol_input): 
    #Reading any file format
    if os.path.isfile(mol_input):
        content_reader = FILE_PARSERS
        mol_format = os.path.splitext(mol_input)[1][1:]
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
    mcs = rdFMCS.FindMCS([query, ref])
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
            rdFMCS.FindMCS([mol, refmol]).smartsString
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

def filtering(mol, receptor, cutoff=1.5, rms=1.0):
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

def cluster(sdf, ref, output):
    import MDAnalysis as mda
    from MDAnalysis.analysis import pca, align
    import matplotlib.pyplot as plt
    import seaborn as sns
    from rpy2 import robjects
    import rpy2.robjects.lib.ggplot2 as ggplot2

    confs = Chem.SDMolSupplier(sdf)

    u = mda.Universe(ref, confs)
    pc = pca.PCA(u, select='all',
                 align=False, mean=None,
                 n_components=None).run()

    backbone = u.select_atoms('all')

    transformed = pc.transform(backbone, n_components=3)
    transformed.shape

    df = pd.DataFrame(transformed,
                      columns=['PC{}'.format(i+1) for i in range(3)])


    df['Index'] = df.index * u.trajectory.dt

    liedata = pd.read_csv('LIE.csv', delimiter=',', header=0)
    PCA_LIE = pd.merge(df, liedata, on='Index')
    PCA_LIE.to_csv('PCA_LIE.csv', index=False, header=True)

    pcs = ['PC1', 'PC2', 'PC3']
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):

            index_data = []

            robjects.r(f'''
            library(ggplot2)

            df <- read.table('PCA_LIE.csv', header=TRUE, sep=",")

            p <- ggplot(df, aes(x={pcs[i]}, y={pcs[j]}, z=LIE)) +
                stat_summary_2d(fun=mean, binwidth = 0.25) 

            plot_data <- ggplot_build(p)
            bin_data <- plot_data$data[[1]]
            write.csv(bin_data, file = '{pcs[i]}_{pcs[j]}.csv', row.names = FALSE)
            ''')

            data = pd.read_csv(f'{pcs[i]}_{pcs[j]}.csv', delimiter=',', header=0)
            raw_data = pd.read_csv('PCA_LIE.csv', delimiter=',', header=0)

            df_sorted = data.sort_values(by='value')
            top_bins = df_sorted.head(10)
            extracted_data = top_bins[['xmin', 'xmax','ymin', 'ymax']]

            for a, rowa in extracted_data.iterrows():
                for b, rowb in raw_data.iterrows():
                    if rowa['xmin'] < rowb[pcs[i]] < rowa['xmax'] and rowa['ymin'] < rowb[pcs[j]] < rowa['ymax']:
                        index_data.append(rowb)
            os.remove(f'{pcs[i]}_{pcs[j]}.csv')

            robjects.r(f'''
            library(ggplot2)

            mydata <- read.table('PCA_LIE.csv', header=TRUE, sep=",")

            p <- ggplot(mydata, aes(x={pcs[i]}, y={pcs[j]})) +
                stat_bin_2d(binwidth = 0.25, aes(fill = after_stat(density)))

            plot_data <- ggplot_build(p)
            bin_data <- plot_data$data[[1]]
            write.csv(bin_data, file = '{pcs[i]}_{pcs[j]}.csv', row.names = FALSE)
            ''')

            data = pd.read_csv(f'{pcs[i]}_{pcs[j]}.csv', delimiter=',', header=0)
            raw_data = pd.read_csv('PCA_LIE.csv', delimiter=',', header=0)

            df_sorted = data.sort_values(by='density')
            top_bins = df_sorted.tail(10)
            extracted_data = top_bins[['xmin', 'xmax','ymin', 'ymax']]

            for a, rowa in extracted_data.iterrows():
                for b, rowb in raw_data.iterrows():
                    if rowa['xmin'] < rowb[pcs[i]] < rowa['xmax'] and rowa['ymin'] < rowb[pcs[j]] < rowa['ymax']:
                        index_data.append(rowb)
            os.remove(f'{pcs[i]}_{pcs[j]}.csv')



    cids = []
    for entry in index_data:
        cids.append(confs[int(entry['Index'])])

    from rdkit.Chem import rdMolAlign
    dists = []
    for i in range(len(cids)):
        for j in range(i):
            dists.append(rdMolAlign.GetBestRMS(cids[i],cids[j]))

    from rdkit.ML.Cluster import Butina
    clusts = Butina.ClusterData(dists, len(cids), 0.75, isDistData=True, reordering=True)
    PC1 = []
    PC2 = []
    PC3 = []
    outf = open('clusts.sdf','a')
    for i in clusts:
        a = index_data[i[0]]
        PC1.append(a['PC1'])
        PC2.append(a['PC2'])
        PC3.append(a['PC3'])
        sdwriter = Chem.SDWriter(outf)
        sdwriter.write(cids[i[0]])
        sdwriter.close()
    outf.close()


    pcx = [PC1, PC2, PC3]
    pcs = ['PC1', 'PC2', 'PC3']
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):

            robjects.r(f'''
            library(ggplot2)
            library(ggdensity)

            df <- read.table('PCA_LIE.csv', header=TRUE, sep=",")
            
            p <- ggplot(df, aes(x={pcs[i]}, y={pcs[j]}, z=LIE)) +
                stat_summary_2d(fun=mean, binwidth = 0.25) +
                scale_fill_gradientn(colours = rainbow(5), limits = c(min(df$LIE),max(df$LIE))) +
                geom_hdr_lines() +
                labs(fill = "LIE") +
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

