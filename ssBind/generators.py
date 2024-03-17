#!/usr/bin/python
import os
import uuid
import itertools
import shutil
import math
import multiprocessing as mp
from rdkit import Chem
from contextlib import closing
from . import chem_tools, plants, rdock, smina, gmx_tools


#Substructure-based alternative BINDing modes generator for protein-ligand systems

class SSBIND:

    def __init__(self, **kwargs):
        self._reference_substructure = kwargs.get('reference_substructure')
        self._query_molecule = kwargs.get('query_molecule')
        self._receptor_file = kwargs.get('receptor_file')
        self._nprocs = kwargs.get('nprocs', mp.cpu_count())
        self._numconf = kwargs.get('numconf', 1000)
        self._curdir = kwargs.get('curdir', os.getcwd())
        self._working_dir = kwargs.get('working_dir', os.path.join(self._curdir, str(uuid.uuid4())))

   
    def generate_conformers(self, generator: str = 'rdkit', degree: float = 60.0, cutoff_dist: float = 1.5, rms: float = 0.2):
        """
        Generates conformers using RDKit or dihedral angle sampling.

        """
        if generator == 'angle':
            molDihedrals = chem_tools.get_uniqueDihedrals(self._reference_substructure, self._query_molecule)
            inputs = itertools.product(chem_tools.degreeRange(degree), repeat=len(molDihedrals))
            if len(molDihedrals) > 3:
                print(f"\nWarning! Too many torsions ({len(molDihedrals)})")
            if len(molDihedrals) > 5:
                print("Exiting due to too many torsions.")
                return
            print(f'\nConformational sampling is running for {len(molDihedrals)} dihedrals.')        
            
            with closing(mp.Pool(processes=self._nprocs)) as pool:
                pool.starmap(chem_tools.gen_conf_angle, [(j, molDihedrals, self._query_molecule, self._reference_substructure) for j in inputs])
        elif generator == 'rdkit':
                #Conformer generation using RDKit.
            with closing(mp.Pool(processes=self._nprocs)) as pool:
                pool.starmap(chem_tools.gen_conf_rdkit, [(self._query_molecule, self._reference_substructure, j) for j in range(self._numconf)])
            
        ###Filter conformers having stearic clashes, clash with the protein, duplicates.
        print('\n{} conformers have been generated.'.format(len(Chem.SDMolSupplier('conformers.sdf', sanitize=False))))
        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(chem_tools.filtering, [(mol, self._receptor_file, cutoff_dist, rms) for i, mol in enumerate(Chem.SDMolSupplier('conformers.sdf', sanitize=False))])

        
    def generate_conformers_plants(self, flexDist = None, flexList = None):
        """
        Generates conformers using PLANTS.

        """       
        output_dir = self._working_dir
        os.makedirs(output_dir)
    
        xyz = " ".join(map(str, [round(coord, 3) for coord in plants.molecule_center(self._query_molecule)]))
        fixedAtom = plants.getAtomConst(self._query_molecule, self._reference_substructure) + 1
        # Handle flexibility for PLANTS
        flex_res = plants.handle_flexibility(flexDist, flexList, self._receptor_file, self._query_molecule, self._reference_substructure)

        plants.SPORES(self._receptor_file, 'receptor.mol2', 'settypes')

        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(plants.plants_docking, [(i, output_dir, 15, 10, xyz, fixedAtom, flex_res) for i in range(math.ceil(self._numconf/10))])
        
        plants.combine_files(output_dir)
        conformers = 'combined_file.pdb' if flex_res else 'conformers.sdf'


    def generate_conformers_rdock(self):
        """
        Generates conformers using rDock.

        """
        rdock_random = os.path.basename(self._working_dir)
        os.makedirs(f'{rdock_random}')

        sdwriter = Chem.SDWriter(f'{rdock_random}/ref.sdf')
        sdwriter.write(self._reference_substructure)
        sdwriter.close()

        rdock.get_tethered(self._reference_substructure, self._query_molecule, rdock_random)
        rdock.prepare_receptor(RECEPTOR_FILE=self._receptor_file, REF_MOL=f'{self._working_dir}/ref.sdf')

        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(rdock.run_rdock, [(i, f'{self._working_dir}/{rdock_random}.sd', f'{rdock_random}') for i in range(math.ceil(self._numconf/10))])

        rdock.combine_files(f'{rdock_random}')
        shutil.rmtree(f'{rdock_random}')

                    
    def run_minimization(self, conformers = 'conformers.sdf', minimizer: str = 'smina', proteinFF: str = 'amber99sb-ildn', FF: str = 'gaff'):
        """
        Performs minimization and scoring.

        """

        conformers_supplier = Chem.SDMolSupplier(conformers, sanitize=False)
        conformers = [(i, mol) for i, mol in enumerate(conformers_supplier)]

        if minimizer == 'gromacs':
            optimize_molecule(conformers)
            gmx_tools.system_setup(self._receptor_file, 'ligand.sdf', proteinFF=args.proteinFF, FF=args.FF)
            trjdir = self._working_dir
            os.makedirs(trjdir)
            with closing(mp.Pool(processes=self._nprocs)) as pool:
                pool.starmap(gmx_tools.minimize, [(i, mol, trjdir) for i, mol in conformers])
        elif minimizer == 'smina':
            conf_dir = self._working_dir
            os.makedirs(conf_dir)
            with closing(mp.Pool(processes=self._nprocs)) as pool:
                pool.starmap(smina.smina_minimize_score, [(i, self._receptor_file, mol, conf_dir) for i, mol in conformers])
            smina.combine_sdf_files('minimized_conformers.sdf', conf_dir, 'Scores.csv')


    def clustering(self, 
                   conformers: str = 'conformers.sdf', 
                   scores: str = 'Scores.csv', 
                   binsize: float = 0.25, 
                   distThresh: float = 0.5, 
                   numbin: int = 10
                   ):    
        """
        Performs clustering based on the conformational distance (RMSD) matrix.

        """

        chem_tools.clustering_poses(
            inputfile=conformers, 
            receptor=self._receptor_file, 
            csv_scores=scores, 
            binsize=binsize, 
            distThresh=distThresh, 
            numbin=numbin, 
            nprocs=self._nprocs
        )

