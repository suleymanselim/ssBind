#!/usr/bin/python
import os, math, shutil, csv, uuid, glob
from rdkit import Chem
from rdkit.Chem.rdMolAlign import AlignMol, GetBestRMS
from rdkit.Chem import AllChem
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import chilife as xl

from chem_tools import MolFromInput

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

            from rdkit.Chem import rdMolAlign
            dists = []
            for i in range(residue.GetNumConformers()):
                for j in range(i):
                    dists.append(rdMolAlign.GetBestRMS(Chem.Mol(residue,confId=i),Chem.Mol(residue,confId=j)))

            from rdkit.ML.Cluster import Butina
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


def get_flex_residues(receptor, ligand, cutoff=3.5): 

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
    from chem_tools import MCS_AtomMap
    AlignMol(ligand, reflig, atomMap=MCS_AtomMap(ligand, reflig))

    pdwriter = Chem.PDBWriter('ligand.pdb', flavor=4)
    pdwriter.write(ligand)
    pdwriter.close()

    SPORES('ligand.pdb', 'ligand.mol2')

    ligcenter = molecule_center(reflig)

    ringAtoms = []
    for pair in MCS_AtomMap(reflig, ligand):
        if any(pair[1] in sublist for sublist in GetRingSystems(ligand, includeSpiro=False)):
            ringAtoms.append(pair[1])

    conf = ligand.GetConformer()
    if not ringAtoms:
        ringAtoms = range(conf.GetNumAtoms())

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

def SPORES(inputfile, outputfile):

    RC = os.system(f'''SPORES --mode complete {inputfile} {outputfile} > SPORES.log 2>&1 ''')
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the SPORES. See the {} for details.'.format(os.path.abspath("SPORES.log\n")))
        

def plants_docking(i, ligand, reflig, receptor, radius=10, output_dir='output', cluster_structures=10, xyz=None, fixedAtom=None, flex_res=None):

    template=(f'''
#Input Options
protein_file 			receptor.mol2

#Binding Site
bindingsite_center {xyz}
bindingsite_radius {radius}

#Cluster Algorithm
cluster_structures {cluster_structures}

#Scoring Functions
scoring_function chemplp
ligand_intra_score lj

#Output Options
write_protein_splitted 1
write_multi_mol2 0


#Fixed Scaffold
ligand_file ligand.mol2 fixed_scaffold_{fixedAtom}


flip_ring_corners 0	

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


        os.makedirs(os.path.join('..', f'docking_conformers'), exist_ok=True)
        for filename in os.listdir('.'):
            if filename.startswith('ligand_entry_00001_conf_') and 'protein' not in filename:
                shutil.move(filename, os.path.join(os.path.join('..', f'docking_conformers'), filename.replace('_entry_00001', f'_entry_{i}')))

        try:
            with open('ranking.csv', 'r') as csv_in, open(os.path.join('..', 'docking_conformers', 'docking_scores.csv'), 'a') as csv_out:
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)

                # Skip the header
                next(reader)

                # Write docking scores
                for row in reader:
                    ligand_entry = row[0]
                    ligand_last = ligand_entry.replace('_entry_00001', f'_entry_{i}')
                    total_score = row[1]
                    writer.writerow([ligand_last, total_score])
            success = True
        except FileNotFoundError:
            print(f"The docking {i} failed. Running again...")

        os.chdir(curdir)
        shutil.rmtree(output_dir)

def filtering(mol, rms=0.2):
    if CheckRMS(mol, rms) == False:
        return
    else:
        outf = open('filtered.sdf','a')
        sdwriter = Chem.SDWriter(outf)
        sdwriter.write(mol)
        sdwriter.close()
        outf.close()

def CheckRMS(moli, rms=1.0):
    ##Filtering identical conformations
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
