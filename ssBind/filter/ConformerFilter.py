import multiprocessing as mp
from contextlib import closing

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdmolops
from rdkit.Chem.Draw import rdMolDraw2D


class ConformerFilter:

    def __init__(self, receptor_file: str, cutoff_dist=1.5, rms=0.2, **kwargs) -> None:
        self._receptor_file = receptor_file
        self._cutoff_dist = cutoff_dist
        self._rms = rms
        self._nprocs = kwargs.get("nprocs", 1)

    def filter_conformers(self, conformers: str = "conformers.sdf") -> None:
        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(
                self._filtering,
                [
                    (mol, self._receptor_file, self._cutoff_dist, self._rms)
                    for _, mol in enumerate(Chem.SDMolSupplier(conformers))
                ],
            )

    @staticmethod
    def _filtering(mol, receptor_file, cutoff_dist, rms):
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

        if ConformerFilter._steric_clash(mol):
            return
        elif ConformerFilter._distance(receptor_file, mol, cutoff_dist):
            return
        elif ConformerFilter._CheckRMS("filtered.sdf", mol, rms):
            return
        else:
            outf = open("filtered.sdf", "a")
            sdwriter = Chem.SDWriter(outf)
            sdwriter.write(mol)
            sdwriter.close()
            outf.close()

    @staticmethod
    def _steric_clash(mol):
        """Identify steric clashes based on mean bond length."""

        ##Identify stearic clashes based on mean bond length
        ditancelist = rdmolops.Get3DDistanceMatrix(mol)[0]
        for i in range(1, len(ditancelist)):
            if ditancelist[i] < 0.5 * rdMolDraw2D.MeanBondLength(mol):
                return True
            else:
                continue
        return False

    @staticmethod
    def _distance(receptor, ligand, cutoff=1.5):
        """Calculate the minimum distance between a protein and a ligand,
        excluding hydrogen atoms, and return True if it's below a cutoff."""

        protein = mda.Universe(receptor)
        ligand = mda.Universe(ligand)

        atom1 = protein.select_atoms("not name H*")
        atom2 = ligand.select_atoms("not name H*")

        distances = mda.analysis.distances.distance_array(
            atom1.positions, atom2.positions
        )

        return distances.min() < cutoff

    @staticmethod
    def _CheckRMS(sdfmol, ref, rms=0.2):
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
            outf = Chem.SDMolSupplier(sdfmol, sanitize=True)
            for i, mol in enumerate(outf):
                if rdMolAlign.GetBestRMS(outf[i], ref) < rms:
                    return True
                else:
                    continue
            return False
        except OSError:
            return False
