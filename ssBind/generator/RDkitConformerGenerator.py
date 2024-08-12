import multiprocessing as mp
from contextlib import closing
from copy import deepcopy
from typing import Dict

from rdkit import Chem
from rdkit.Chem.AllChem import EmbedMolecule
from rdkit.Chem.rdchem import Mol

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
        """Generate conformers using random embeddings via RDKit."""

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
        """Generate one conformer using RDKit.

        Args:
            seed (int): Random seed for constrained embedding.
        """
        ligand = Chem.AddHs(self._query_molecule)
        ligEmbed = self._embed(ligand, seed + 1)
        ligMin = self._minimize(ligEmbed)
        outmol = Chem.RemoveHs(ligMin)

        with open("conformers.sdf", "a") as outf:
            with Chem.SDWriter(outf) as sdwriter:
                sdwriter.write(outmol)

    def _embed(self, ligand: Mol, seed: int = -1) -> Mol:
        """Use distance geometry (RDKit EmbedMolecule) to generate a conformer of ligand
        tethering to the reference structure (coordMap).

        Args:
            ligand (Mol): Molecule to generate conformer
            seed (int, optional): Random seed for embedding. Defaults to -1.

        Returns:
            Mol: New conformer of ligand
        """
        coordMap = {}
        ligConf = ligand.GetConformer(0)
        for _, ligIdx in self._mappingRefToLig:
            ligPtI = ligConf.GetAtomPosition(ligIdx)
            coordMap[ligIdx] = ligPtI

        l_embed = deepcopy(ligand)
        EmbedMolecule(
            l_embed, coordMap=coordMap, randomSeed=seed, useExpTorsionAnglePrefs=False
        )
        self._alignToRef(l_embed)
        return l_embed
