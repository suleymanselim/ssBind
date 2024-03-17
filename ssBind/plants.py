#!/usr/bin/python

# Standard library imports
import os
import math
import shutil, csv, uuid, glob, re

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdmolops
from rdkit.Chem.rdMolAlign import AlignMol, GetBestRMS
from rdkit.ML.Cluster import Butina

# MDAnalysis imports 
import MDAnalysis as mda
from MDAnalysis.analysis import distances

# Custom utility imports
from .chem_tools import MolFromInput, obabel_convert, MCS_AtomMap
import chilife as xl

# Data handling 
import pandas as pd


def get_close_residues(receptor, ligand, cutoff=3.5): 


    protein = mda.Universe(receptor)
    ligand = mda.Universe(ligand)

    atom1 = protein.select_atoms("not name H* CA C N O")
    atom2 = ligand.select_atoms("not name H*")

    res_list = []
    for res in atom1.residues:
        resi = atom1.select_atoms("resid {} ".format(res.resid))
        distances = mda.analysis.distances.distance_array(resi.positions, atom2.positions)
        if distances.min() < cutoff:
            if res.resname == 'ALA':
                continue
            res_list.append(res)
            SL1 = xl.RotamerEnsemble(res.resname, res.resid, protein=protein, eval_clash = True, dihedral_sigmas=360, sample = 1000)
            xl.save('test_{}_{}.pdb'.format(res.resname,res.resid), SL1)

            residue = MolFromInput('test_{}_{}.pdb'.format(res.resname,res.resid))

            
            dists = []
            for i in range(residue.GetNumConformers()):
                for j in range(i):
                    dists.append(rdMolAlign.GetBestRMS(Chem.Mol(residue,confId=i),Chem.Mol(residue,confId=j)))

            
            rmsd_cutoff = 0.5
            for _ in range(10):
                clusts = Butina.ClusterData(dists, residue.GetNumConformers(), rmsd_cutoff, isDistData=True, reordering=True)
                if len(clusts) > 5:
                    rmsd_cutoff += 0.1
                    del(clusts)
                else:
                    break

            outf = open('file_{}_{}.pdb'.format(res.resname,res.resid),'a')
            for i in clusts:
                pdwriter = Chem.PDBWriter(outf, flavor=1)
                pdwriter.write(Chem.Mol(residue,confId=i[0]))
            pdwriter.close()
            outf.close()
            #os.remove('test_{}_{}.pdb'.format(res.resname,res.resid))

# Calculate the center of the molecule
def molecule_center(mol):

    conf = mol.GetConformer()
    center = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    center = [sum(coord.x for coord in center) / len(center),
              sum(coord.y for coord in center) / len(center),
              sum(coord.z for coord in center) / len(center)]

    return center


def get_flex_residues(receptor, ligand, reference, cutoff=3.5): 
    
    
    AlignMol(ligand, reference, atomMap=MCS_AtomMap(ligand, reference))
    
    protein = mda.Universe(receptor)
    ligand = mda.Universe(ligand)

    atom1 = protein.select_atoms("not name H* CA CB C N O")
    atom2 = ligand.select_atoms("not name H*")

    res_list = []
    for res in atom1.residues:
        resi = atom1.select_atoms("resid {} ".format(res.resid))
        distances = mda.analysis.distances.distance_array(resi.positions, atom2.positions)
        if distances.min() < cutoff:
            res_list.append([res.resname,res.resid])
    return res_list

def getAtomConst(ligand, reflig):
    
    AlignMol(ligand, reflig, atomMap=MCS_AtomMap(ligand, reflig))

    pdwriter = Chem.PDBWriter('ligand.pdb', flavor=4)
    pdwriter.write(ligand)
    pdwriter.close()

    SPORES('ligand.pdb', 'ligand.mol2', 'complete')

    ligcenter = molecule_center(reflig)

    ringAtoms = []
    for pair in MCS_AtomMap(reflig, ligand):
        if any(pair[1] in sublist for sublist in GetRingSystems(ligand)):
            ringAtoms.append(pair[1])
    

    conf = ligand.GetConformer()
    if not ringAtoms:
        ringAtoms = [t[1] for t in MCS_AtomMap(reflig, ligand)]

    dis = []
    for atom in ringAtoms:
        dis.append({atom : math.sqrt((ligcenter[0] - conf.GetAtomPosition(atom).x)**2 + (ligcenter[1] - conf.GetAtomPosition(atom).y)**2 + (ligcenter[2] - conf.GetAtomPosition(atom).z)**2)})
    min_key = min(dis, key=lambda x: list(x.values())[0])

    atomID = list(min_key.keys())[0]
    return atomID


def GetRingSystems(mol, includeSpiro=False): #https://gist.github.com/greglandrum/de1751a42b3cae54011041dd67ae7415
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon>1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return systems

def SPORES(inputfile, outputfile, mode):

    RC = os.system(f'''SPORES --mode {mode} {inputfile} {outputfile} > SPORES.log 2>&1 ''')
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the SPORES. See the {} for details.'.format(os.path.abspath("SPORES.log\n")))
        

def plants_docking(i, dock_dir, radius=15, cluster_structures=10, xyz=None, fixedAtom=None, flex_res=None):

    template=(f'''
#Input Options
protein_file             receptor.mol2

#Binding Site
bindingsite_center {xyz}
bindingsite_radius {radius}

#Cluster Algorithm
cluster_structures {cluster_structures}

#Scoring Functions
scoring_function chemplp
ligand_intra_score lj

#Output Options
write_multi_mol2 0

#Fixed Scaffold
ligand_file ligand.mol2 fixed_scaffold_{fixedAtom}

#Flexible Side-chains
''')

    success = False

    while not success:
        f = open("plants_config", "w")
        f.write(template)
        for item in flex_res:
            f.write(item + '\n')
        f.close()

        curdir = os.getcwd()
        output_dir = str(uuid.uuid4())
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(os.path.join(curdir, 'receptor.mol2'), output_dir)
        shutil.copy(os.path.join(curdir, 'ligand.mol2'), output_dir)
        shutil.copy(os.path.join(curdir, 'plants_config'), output_dir)
        os.chdir(output_dir)

        os.system(f'''PLANTS --mode screen plants_config > PLANTS.log 2>&1 ''')

        for filename in os.listdir('.'):
            if flex_res == []:
                if filename.startswith('ligand_entry_00001_conf_') and 'protein' not in filename:
                    shutil.move(filename, os.path.join(os.path.join('..', str(dock_dir)), filename.replace('_entry_00001', f'_{i}')))

            else:
                if filename.startswith('ligand_entry_00001_conf_') and 'protein' not in filename:
                    filenamepdb = filename.replace('_entry_00001', f'_{i}')
                    obabel_convert(filename, 'ligand.pdb')
                    obabel_convert(filename.replace('.mol2', '_protein.mol2'), 'protein.pdb')
                    mol1 = Chem.MolFromPDBFile('ligand.pdb', removeHs=False, sanitize=False)
                    mol2 = Chem.MolFromPDBFile('protein.pdb', removeHs=False, sanitize=False)
                    mol = rdmolops.CombineMols(mol1, mol2)
                    Chem.MolToPDBFile(mol, "complex.pdb", flavor=1)
                    shutil.move("complex.pdb", os.path.join('..', dock_dir, filenamepdb.replace('.mol2', '.pdb')))

        try:
            with open('ranking.csv', 'r') as csv_in, open(os.path.join('..', dock_dir, 'Scores.csv'), 'a') as csv_out:
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)

                # Skip the header
                next(reader)

                # Write docking scores
                for row in reader:
                    ligand_entry = row[0]
                    ligand_last = ligand_entry.replace('_entry_00001', f'_{i}')
                    total_score = row[1]
                    writer.writerow([ligand_last, total_score])
            success = True
        except FileNotFoundError:
            print(f"The docking {i} failed. Running again...")

        os.chdir(curdir)
        shutil.rmtree(output_dir)

def filtering(mol, numconf, rms=0.1):
    ##Filtering identical conformations
    if CheckRMS(mol, rms):
        outf = open('filtered.sdf', 'a')
        sdwriter = Chem.SDWriter(outf)
        sdwriter.write(mol)
        sdwriter.close()
        outf.close()
    else:
        return

def CheckRMS(moli, rms=0.1):
    try:
        outf = Chem.SDMolSupplier('filtered.sdf')
        for i, mol in enumerate(outf):
            rms_calc = GetBestRMS(mol, moli)
            if rms_calc < rms:
                return False
            else:
                continue
        return True
    except OSError:
        return True  

def natural_sort_key(s):
    """
    Generate a key for sorting strings that contain numbers,
    ensuring that numerical parts are sorted numerically.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def combine_files(dockdir):

    # Directory containing the Mol2 files
    files = sorted([f for f in os.listdir(dockdir) if f.endswith('.mol2')], key=natural_sort_key)

    
    csv_path = f'{dockdir}/Scores.csv'
    df = pd.read_csv(csv_path, header=None)    
    
    flex = False
    if len(files) == 0:
        flex = True
        files = sorted([f for f in os.listdir(dockdir) if f.endswith('.pdb')], key=natural_sort_key)
        combined_lines = []
        conect_lines = []
        model_count = 0
        for pdb_file in files:
            with open(f'{dockdir}/{pdb_file}', 'r') as file:
                lines = file.readlines()
                atom_lines = [line for line in lines if line.startswith('ATOM')]
                if pdb_file == files[0]: 
                    conect_lines = [line for line in lines if line.startswith('CONECT')]
                    shutil.copy(f'{dockdir}/{pdb_file}', 'ref_conf.pdb')
                combined_lines.append(f"MODEL        {model_count}\n")
                combined_lines.extend(atom_lines)
                combined_lines.append("ENDMDL\n")  # Append ENDMDL after each file's ATOM lines
                model_count += 1
        combined_lines.extend(conect_lines)
        combined_lines.append("END\n")
        with open('combined_file.pdb', 'w') as combined_file:
           combined_file.writelines(combined_lines)
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x + '.pdb')
    else:
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x + '.mol2')

        # Create an SDF writer
    sdf_writer = Chem.SDWriter('conformers.sdf')

        # Process each Mol2 file
    for f in files:
        path = os.path.join(dockdir, f)
        if flex:
            mol = Chem.MolFromPDBFile(path, sanitize=False)
        else:
            mol = Chem.MolFromMol2File(path, sanitize=False)
        if mol is not None:
            sdf_writer.write(mol)
    sdf_writer.close()


    # Create a dictionary for mapping filenames to their rows
    filename_to_row = {filename: row for row, filename in enumerate(files)}

    # Sort the dataframe based on the order of the mol2 files
    df['sort_index'] = df.iloc[:, 0].map(filename_to_row)
    df.sort_values(by='sort_index', inplace=True)

    df.drop(columns=['sort_index'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save the rearranged CSV with index column
    df.columns = ['names', 'Score']
    df['names'] = df['names'].str.replace(r'\.\w+$', '', regex=True)

    df.to_csv('Scores.csv', index=True, index_label='Index')
    shutil.rmtree(dockdir)

def handle_flexibility(flexDist = None, flexList = None, receptor_file = 'receptor.pdb', query_molecule = 'ligand.mol2', reference_substructure = 'reference.mol2'):
    # Handle flexibility for PLANTS
    if flexDist is not None:
        return [f"flexible_protein_side_chain_string {residue[0]}{residue[1]}" for residue in plants.get_flex_residues(
                                                                                                          receptor_file, 
                                                                                                          query_molecule, 
                                                                                                          reference_substructure, 
                                                                                                          flexDist)]
    elif flexList is not None:
        residuelist = [[item[:3], int(item[3:])] for item in flexList.split(',')] 
        return [f"flexible_protein_side_chain_string {residue[0]}{residue[1]}" for residue in residuelist]
    else:
        return []    
