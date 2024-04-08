#!/usr/bin/python

# Standard library imports
import os
import math
import logging
from copy import deepcopy

# Third-party imports for data handling
import pandas as pd
import numpy as np

# RDKit imports 
try:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS, rdMolAlign, rdMolTransforms, AllChem
    from rdkit.Chem import rdmolops
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Geometry import Point3D
    from rdkit.ML.Cluster import Butina
    from rdkit.Chem.rdmolfiles import *
except ImportError as e:
    logging.error(f"Failed to import RDKit: {e}")


# MDAnalysis imports 
import MDAnalysis as mda
from MDAnalysis.analysis import pca, align, distances

# Rpy2 imports for R integration
try:
    from rpy2.robjects import r, pandas2ri
    pandas2ri.activate()
    import rpy2.robjects.lib.ggplot2 as ggplot2
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
    rpy2_logger.setLevel(logging.ERROR)
except ImportError as e:
    logging.error(f"Failed to import rpy2: {e}")


# OpenBabel imports 
try:
    from openbabel import openbabel, OBConversion, vector3
except ImportError as e:
    logging.error(f"Failed to import OpenBabel: {e}")


from .gmx_tools import gmx_grompp
import multiprocessing as mp
from contextlib import closing

from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper


def is_file(fname):
    """
    Check if a given path points to an existing file.

    Returns:
    - True if the path points to an existing file, False otherwise.
    """
    return os.path.isfile(fname)


def which(program):
    """
    Search for an executable in the system's PATH.

    Returns:
    - The full path to the executable if found; None otherwise.
    """
    def is_exe(fpath):
        """Check if a given path is an executable file."""
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
    
    
def MolFromInput(molecule): #The code was adapted from https://pypi.org/project/rdkit-utilities/
    """
    Create a RDKit molecule object from an input file of a supported format.

    Raises:
    - SystemExit: If the input file does not exist.
    - TypeError: If the molecule format is not supported or the molecule cannot be created.
    """
    
    FILE_PARSERS = {
    "mol": MolFromMolFile,
    "mol2": MolFromMol2File,
    "pdb": MolFromPDBFile,
    "sdf": MolFromMolFile,
    }
    # Check if the input is a file
    if not is_file(molecule):
        raise SystemExit(f'\nERROR! Input file ({molecule}) is not found!!!')

    content_reader = FILE_PARSERS
    mol_format = os.path.splitext(molecule)[1][1:].lower()

    # Attempt to read the file with a specified format
    if mol_format:
        try:
            reader = content_reader[mol_format]
            return reader(molecule)
        except KeyError:
            supported_formats = ", ".join(FILE_PARSERS)
            raise TypeError(f"Molecule format {mol_format} not supported. Supported formats: {supported_formats}")

    # Fallback: try reading the file with all supported formats
    for reader in content_reader.values():
        try:
            mol = reader(molecule)
            if mol is not None:
                return mol
        except RuntimeError:
            continue

    # If no reader was successful, raise an error
    supported_formats = ", ".join(FILE_PARSERS)
    raise TypeError(f"Could not create an RDKit molecule from {molecule}. Try passing in a `mol_format`. Supported formats: {supported_formats}")


def replace_coor(input_sdf, input_mol2, output): 
    """
    Replace coordinates from an SDF file to a MOL2 file.
    
    Parameters:
    - input_sdf: Path to the input SDF file.
    - input_mol2: Path to the input MOL2 file.
    - output: Path for the output MOL2 file with updated coordinates.
    """
    # Stop logging to avoid clutter
    openbabel.obErrorLog.StopLogging()

    # Setup for reading SDF file
    sdf_conv = OBConversion()
    sdf_conv.SetInFormat("sdf")
    mol_sdf = openbabel.OBMol()
    if not sdf_conv.ReadFile(mol_sdf, input_sdf):
        raise IOError(f"Failed to read SDF file: {input_sdf}")

    # Setup for reading and writing MOL2 file
    mol2_conv = OBConversion()
    mol2_conv.SetInAndOutFormats("mol2", "mol2")
    mol_mol2 = openbabel.OBMol()
    if not mol2_conv.ReadFile(mol_mol2, input_mol2):
        raise IOError(f"Failed to read MOL2 file: {input_mol2}")

    # Replace coordinates
    for atom_sdf, atom_mol2 in zip(openbabel.OBMolAtomIter(mol_sdf), openbabel.OBMolAtomIter(mol_mol2)):
        atom_mol2.SetVector(vector3(atom_sdf.GetX(), atom_sdf.GetY(), atom_sdf.GetZ()))

    # Update residue and title information
    residue = next(openbabel.OBResidueIter(mol_mol2))
    residue.SetName("LIG")
    mol_mol2.SetTitle("LIG")
    
    # Write the modified MOL2 file
    if not mol2_conv.WriteFile(mol_mol2, output):
        raise IOError(f"Failed to write MOL2 file: {output}")

    
def obabel_convert(input_file, output_file, resname: str = None, ph: float = None, uniqueNames: bool = False):
    """
    Converts molecular file formats using Open Babel, with options for setting residue names,
    adjusting hydrogens for pH, de-aromatizing molecules, and ensuring unique atom names.

    Parameters:
    - input_file: Path to the input file.
    - output_file: Path for the output file.
    - resname: Optional residue name to set for the molecule.
    - ph: Optional pH level to correct hydrogen atoms for specific pH.
    - uniqueNames: If True, ensures that atom names are unique.
    """
    openbabel.obErrorLog.StopLogging()

    conv = openbabel.OBConversion()
    input_format = input_file.split('.')[-1].lower()
    output_format = output_file.split('.')[-1].lower()
    
    conv.SetInAndOutFormats(input_format, output_format)
    mol = openbabel.OBMol()
    
    if not conv.ReadFile(mol, input_file):
        raise IOError(f"Failed to read input file: {input_file}")

    if resname is not None:
    	for residue in openbabel.OBResidueIter(mol):
    	    residue.SetName(resname)
    	mol.SetTitle(resname)
    	
    if ph is not None:
        mol.AddHydrogens(False, True, ph)
    
    for bond in openbabel.OBMolBondIter(mol):
        bond.SetAromatic(False)

    for atom in openbabel.OBMolAtomIter(mol):
        atom.SetAromatic(False)
          
    if not conv.WriteFile(mol, output_file):
        raise IOError(f"Failed to write output file: {output_file}")

    if uniqueNames:
        makeUniqueNames(output_file)
        pass

    
def makeUniqueNames(mol2_file):
    """
    Ensures that each atom in a MOL2 file has a unique name by appending a number to duplicate atom names.

    Parameters:
    - mol2_file: Path to the MOL2 file to be processed.
    """
    number = {}
    updated_lines = []

    with open(mol2_file, 'r') as file:
        lines = file.readlines()

    processing_atoms_section = False
    for line in lines:
        if line.startswith('@<TRIPOS>ATOM'):
            updated_lines.append(line)
            processing_atoms_section = True
        elif line.startswith('@<TRIPOS>') and processing_atoms_section:
            updated_lines.append(line)
            processing_atoms_section = False
        elif processing_atoms_section:
            atom_data = line.split()
            atom_name = atom_data[1]
            if atom_name not in number:
                number[atom_name] = 1
            else:
                number[atom_name] += 1
            atom_data[1] = f"{atom_name}{number[atom_name]}"
            atom_data[7] = 'LIG'
            formatted_line = '{0:6d} {1:<8s} {2:10.4f} {3:10.4f} {4:10.4f} {5:<8s} {6:6d} {7:<8s} {8:10.6f}\n'.format(
                int(atom_data[0]), atom_data[1], float(atom_data[2]), float(atom_data[3]), float(atom_data[4]),
                atom_data[5], int(atom_data[6]), atom_data[7], float(atom_data[8]))
            updated_lines.append(formatted_line)
        else:
            updated_lines.append(line)

    # Rewrite the file with updated lines
    with open(mol2_file, 'w') as file:
        file.writelines(updated_lines)


def getDihedralMatches(mol): #The code was adapted from David Koes https://github.com/dkoes/rdkit-scripts/blob/master/rdallconf.py
    """
    Get dihedral matches in a molecule.
    
    Parameters:
    - mol: An RDKit molecule object.

    Returns:
    - A list of tuples, each representing the atom indices of a dihedral match.
    """
    pattern = r"*~[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]~*"
    qmol = Chem.MolFromSmarts(pattern)
    matches = mol.GetSubstructMatches(qmol)
    
    uniqmatches = []
    seen = set()
    for a, b, c, d in matches:
        if (b, c) not in seen:
            seen.add((b, c))
            uniqmatches.append((a, b, c, d))
    return uniqmatches


def MCS_AtomMap(query, ref):
    """
    Build an atom mapping based on the Maximum Common Substructure (MCS) between two molecules.

    Parameters:
    - query: The query RDKit molecule object.
    - ref: The reference RDKit molecule object.

    Returns:
    - A list of tuples, each representing a pair of atom indices from the query and reference molecules
      that correspond to each other in the MCS.
    """
    mcs = rdFMCS.FindMCS([query, ref], completeRingsOnly=True, matchValences=True)
    submol = Chem.MolFromSmarts(mcs.smartsString)
    refMatch = ref.GetSubstructMatch(submol)
    queryMatch = query.GetSubstructMatch(submol)
    Amap = []
    for i in range(len(refMatch)):
        Amap.append((queryMatch[i], refMatch[i]))
    return Amap
    

def get_uniqueDihedrals(refmol, mol):
    """
    Identify unique dihedral matches in a molecule that are not found in the MCS with a reference molecule.

    Parameters:
    - refmol: The reference RDKit molecule object.
    - mol: The RDKit molecule object to search for unique dihedrals.

    Returns:
    - A list of tuples, each representing the atom indices of a dihedral unique to `mol` and not part of its MCS with `refmol`.
    """
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
    """
    Produces a list of angle radians for dihedral sampling within a 360-degree range.

    Parameters:
    - inc: The increment in degrees for each step in the range.

    Returns:
    - A list of radians corresponding to the degree increments within a 360-degree range.
    """
    degrees = []
    deg = 0
    while deg < 360.0:
        rad = math.pi*deg / 180.0
        degrees.append(rad)
        deg += inc
    return degrees


def gen_conf_angle(j, mol_Dihedrals, mol, reflig):
    """
    Generate a conformer using Dihedral angles.
    """
    intD = 0
    for i in mol_Dihedrals:
        rdMolTransforms.SetDihedralRad(mol.GetConformer(),*i,value=j[intD])
        intD += 1
    rdMolAlign.AlignMol(mol, reflig, atomMap=MCS_AtomMap(mol, reflig))
    outf = open('conformers.sdf','a')
    sdwriter = Chem.SDWriter(outf)
    sdwriter.write(mol)
    sdwriter.close()
    outf.close()

def gen_conf_rdkit(mol: Chem.rdchem.Mol, ref_mol: Chem.rdchem.Mol, j):
    """
    Generate a conformer using RDKit.
    
    Parameters:
    - mol: The input molecule (RDKit Mol object).
    - ref_mol: The reference molecule for generating the core (RDKit Mol object).
    - j: Random seed for constrained embedding.
    """
    ref_smi = Chem.MolToSmiles(Chem.MolFromSmarts(rdFMCS.FindMCS([mol, ref_mol], completeRingsOnly=True, matchValences=True).smartsString))
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
    """
    Check if any conformation in an SDF file is identical to a reference molecule based on RMSD.
    Calculates the optimal RMS for aligning two molecules, taking symmetry into account. 

    Parameters:
    - sdfmol: Path to the SDF file containing molecule conformations.
    - ref: The reference RDKit molecule object.
    - rms: The RMSD threshold for considering two conformations identical. Default is 0.2.

    Returns:
    - True if any conformation's RMSD to the reference is below the threshold, False otherwise.
    """
    try:
        outf = Chem.SDMolSupplier(sdfmol, sanitize=False)
        for i, mol in enumerate(outf):
            if rdMolAlign.GetBestRMS(outf[i], ref) < rms:
                return True
            else:
                continue
        return False
    except OSError:
        return False 

    
def steric_clash(mol):
    """Identify steric clashes based on mean bond length."""
    
    ##Identify stearic clashes based on mean bond length
    ditancelist = rdmolops.Get3DDistanceMatrix(mol)[0]
    for i in range(1, len(ditancelist)):
        if ditancelist[i] < 0.5 * rdMolDraw2D.MeanBondLength(mol):
            return True
        else:
            continue
    return False

def distance(receptor, ligand, cutoff=1.5): 
    """Calculate the minimum distance between a protein and a ligand,
       excluding hydrogen atoms, and return True if it's below a cutoff."""

    protein = mda.Universe(receptor)
    ligand = mda.Universe(ligand)

    atom1 = protein.select_atoms("not name H*")
    atom2 = ligand.select_atoms("not name H*")

    distances = mda.analysis.distances.distance_array(atom1.positions, atom2.positions)

    return distances.min() < cutoff


def filtering(mol, receptor, cutoff=1.5, rms=0.2):
    """
    Filters out molecules based on steric clash, distance to the receptor,
    and RMS criteria, and appends those passing all checks to 'filtered.sdf'.

    The function checks if a molecule should be filtered out based on three criteria:
    - Having a steric clash.
    - Being within a certain cutoff distance to a receptor.
    - Having an RMS value below a specified rms when compared to molecules.

    Parameters:
    - mol: The molecule to be checked and possibly written to 'filtered.sdf'.
    - receptor: The receptor molecule used for distance comparison.
    - cutoff (float): The distance cutoff for filtering based on proximity to the receptor. 
    - rms (float): The RMS cutoff for filtering based on RMS comparison to existing entries in 'filtered.sdf'.

    Returns:
    - None: The function returns nothing. It writes molecules passing all filters to 'filtered.sdf'.
    """

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

# Function to calculate RMSD, designed to be compatible with multiprocessing.Pool
def calculate_rms(params):
    i, j, cid_i, cid_j = params
    mol1 = Molecule.from_rdkit(cid_i)
    mol2 = Molecule.from_rdkit(cid_j)
    rms = rmsdwrapper(mol1, mol2)
    #rms = GetBestRMS(cid_i, cid_j)
    return i, j, rms
                    
def clustering_poses(inputfile, receptor, csv_scores, binsize, distThresh, numbin, nprocs):
    """
    Performs clustering analysis on the PCA-transformed data.

    """
    
    input_format = inputfile.split('.')[-1].lower()
    
    flex = False
    if input_format == 'xtc':
        
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
    elif input_format == 'pdb':
        confs = Chem.SDMolSupplier('conformers.sdf')
        u = mda.Universe(confs[0], confs)        
        flex = True
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

            r(f'''
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

            r(f'''
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


    #params = [(i, j, cids) for i in range(len(cids)) for j in range(i+1, len(cids))]

    #with closing(mp.Pool(processes=nprocs)) as pool:
    #    dists = pool.map(calculate_best_rms, params)

    dists = []    
    tasks = [(i, j, cids[i], cids[j]) for i in range(len(cids)) for j in range(i)]
    with closing(mp.Pool(processes=nprocs)) as pool:
        results = pool.map(calculate_rms, tasks)
    
    dists = [rms[0] for _, _, rms in sorted(results, key=lambda x: (x[0], x[1]))]
    
    #for i in range(len(cids)):
    #    for j in range(i):
            #mol1 = Molecule.from_rdkit(cids[i])
            #mol2 = Molecule.from_rdkit(cids[j])
            #rms = rmsdwrapper(mol1, mol2)
    #        rms = rdMolAlign.GetBestRMS(cids[i],cids[j])
    #        dists.append(rms)

    clusts = Butina.ClusterData(dists, len(cids), distThresh, isDistData=True, reordering=True)
        
    #from sklearn.cluster import DBSCAN
    #from scipy.spatial.distance import squareform
    #dbscan = DBSCAN(metric='precomputed', eps=0.75, min_samples=5, algorithm = 'auto', n_jobs = 8 )
    #clustering = dbscan.fit(squareform(dists))

    
    PC1 = []
    PC2 = []
    PC3 = []
    mode = 1
    for i in clusts:
        a = index_data[i[0]]
        PC1.append(a['PC1'])
        PC2.append(a['PC2'])
        PC3.append(a['PC3'])
        dict_i = index_dict[i[0]]
        
        if flex == True:
           model = int(next(iter(dict_i.values())))
           get_model_compex(inputfile, model, receptor, f'model_{mode}.pdb')
        else:
            sdwriter = Chem.SDWriter(f'model_{mode}.sdf')
            sdwriter.write(confs[int(next(iter(dict_i.values())))])
            sdwriter.close()
        mode += 1
    


    pcx = [PC1, PC2, PC3]
    pcs = ['PC1', 'PC2', 'PC3']
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):

            r(f'''
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

            ggsave("{pcs[i]}-{pcs[j]}.svg", width = 7, height = 7)

            ''')

	
def find_nearest_conf_to_average(input_file):
    """
    Finds the index of the conformation closest to the average atomic positions 
    in a trajectory.

    This function calculates the average atomic positions across all conformations
    in a trajectory and identifies the conformation that is closest to this average
    configuration based on RMSD (Root Mean Square Deviation).

    """

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


def parse_pdb_line(line):
    """Parse a line of PDB file to extract atom information and coordinates."""
    fields = {
        'record': line[:6].strip(),
        'atom_serial_number': int(line[6:11].strip()),
        'atom_name': line[12:16].strip(),
        'alternate_location_indicator': line[16].strip(),
        'residue_name': line[17:20].strip(),
        'chain_identifier': line[21].strip(),
        'residue_sequence_number': int(line[22:26].strip()),
        'code_for_insertions': line[26].strip(),
        'x': float(line[30:38]),
        'y': float(line[38:46]),
        'z': float(line[46:54]),
        'occupancy': float(line[54:60].strip()),
        'temp_factor': float(line[60:66].strip()),
        'element_symbol': line[76:78].strip(),
        'charge': line[78:80].strip()
    }
    return fields


def get_model_compex(confs, model, source_file, output):
    """
    Extracts a specific model from a PDB file and updates the coordinates of atoms
    in a source PDB file based on this model.

    Args:
        confs (str): Path to the PDB file containing multiple models from which to extract.
        model (int): The model number to extract and use for coordinate updates.
        source_file (str): Path to the source PDB file whose atom coordinates will be updated.
        output (str): Path to the output file where the updated PDB information will be written.

    """
    write = False
    model_lines = []
    with open(confs, 'r') as f_in:
        models = f_in.readlines()
        for line in models:
            if line.startswith('MODEL') and str(line.split()[1]) == str(model):
                write = True
                model_lines.append(line)
            if line.startswith('ATOM') and write:
                model_lines.append(line)
            if line.startswith('ENDMDL') and write:
                model_lines.append(line)
                write = False
            if line.startswith('CONECT'):
                model_lines.append(line)
        model_lines.append('END')

    """Replace coordinates in source atoms list with those from target atoms list based on matching criteria."""
    with open(source_file, 'r') as sfile:
        source_lines = sfile.readlines()
    
    source_atoms = [parse_pdb_line(line) for line in source_lines if line.startswith(("ATOM", "HETATM"))]
    model_atoms = [parse_pdb_line(line) for line in model_lines if line.startswith(("ATOM", "HETATM"))]
    
    lig_atoms = [atom for atom in model_atoms if atom['residue_name'] == 'UNL']

    for source_atom in source_atoms:
        for model_atom in model_atoms:
            if (source_atom['atom_name'] == model_atom['atom_name'] and
                source_atom['residue_name'] == model_atom['residue_name'] and
                source_atom['residue_sequence_number'] == model_atom['residue_sequence_number']):
                source_atom['x'], source_atom['y'], source_atom['z'] = model_atom['x'], model_atom['y'], model_atom['z']
                break
    with open(output, 'w') as file:
        for atom in source_atoms:
            file.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                atom['record'], atom['atom_serial_number'], atom['atom_name'], atom['alternate_location_indicator'],
                atom['residue_name'], atom['chain_identifier'], atom['residue_sequence_number'], atom['code_for_insertions'],
                atom['x'], atom['y'], atom['z'], atom['occupancy'], atom['temp_factor'], atom['element_symbol'], atom['charge']))
        for atom in lig_atoms:
            file.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                atom['record'], atom['atom_serial_number'], atom['atom_name'], atom['alternate_location_indicator'],
                atom['residue_name'], atom['chain_identifier'], atom['residue_sequence_number'], atom['code_for_insertions'],
                atom['x'], atom['y'], atom['z'], atom['occupancy'], atom['temp_factor'], atom['element_symbol'], atom['charge']))


def optimize_molecule(input_file='conformers.sdf', output_file='ligand.sdf'):
    """
    Optimizes the geometry of a molecule loaded from an SDF file and saves the result to another SDF file.

    Parameters:
    - input_file (str): The path to the input SDF file containing many conformers.
    - output_file (str): The path to the output SDF file where the optimized molecule will be saved.
    """
    # Load molecule from SDF file
    sdf = MolFromInput(input_file)

    writer = Chem.SDWriter(output_file)
    mol = Chem.AddHs(sdf, addCoords=True)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
    mol.SetProp('_Name', 'LIG')
    writer.write(mol)
    writer.close()      

    # Set molecule name
    

