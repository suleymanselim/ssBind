import multiprocessing as mp
from contextlib import closing
from copy import deepcopy
from typing import Dict

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

from ssBind.generator import AbstractConformerGenerator


class RDkitConformerGenerator(AbstractConformerGenerator):

    def __init__(
        self,
        query_molecule: str,
        reference_substructure: str,
        receptor_file: str,
        **kwargs: Dict
    ) -> None:
        super().__init__(
            query_molecule, reference_substructure, receptor_file, **kwargs
        )

    def generate_conformers(self) -> None:

        # Conformer generation using RDKit.
        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(self._gen_conf_rdkit, [(j,) for j in range(self._numconf)])

        ###Filter conformers having stearic clashes, clash with the protein, duplicates.
        print(
            "\n{} conformers have been generated.".format(
                len(Chem.SDMolSupplier("conformers.sdf"))
            )
        )

    def _gen_conf_rdkit(self, seed: int) -> None:
        """
        Generate a conformer using RDKit.

        Parameters:
        - mol: The input molecule (RDKit Mol object).
        - ref_mol: The reference molecule for generating the core (RDKit Mol object).
        - j: Random seed for constrained embedding.
        """
        ligand = Chem.AddHs(self._query_molecule)
        ligEmbed = self._embed(ligand, seed + 1)
        ligMin = self._minimize(ligEmbed)
        outmol = Chem.RemoveHs(ligMin)

        with open("conformers.sdf", "a") as outf:
            with Chem.SDWriter(outf) as sdwriter:
                sdwriter.write(outmol)
