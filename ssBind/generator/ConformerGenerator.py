from abc import abstractmethod
from copy import deepcopy
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.rdchem import Mol


class ConformerGenerator:

    def __init__(
        self, query_molecule: str, reference_substructure: str, **kwargs
    ) -> None:
        self._query_molecule = query_molecule
        self._reference_substructure = reference_substructure
        self._nprocs = kwargs.get("nprocs", 1)
        self._numconf = kwargs.get("numconf", 1)
        distTol = kwargs.get("distTol", 1.0)

        self._mappingRefToLig = self._MCS_AtomMap(
            query_molecule, reference_substructure, distTol
        )
        self._mappingLigToRef = [(j, i) for i, j in self._mappingRefToLig]

        fixed_atoms = list(zip(*self._mappingLigToRef))[0]
        self._query_molecule.SetProp("fixed_atoms", str(fixed_atoms))

    @abstractmethod
    def generate_conformers(self) -> None:
        pass

    @staticmethod
    def _MCS_AtomMap(ligand: Mol, ref: Mol, distTol: float = 1.0) -> List[Tuple[int]]:

        mcs = rdFMCS.FindMCS([ligand, ref], completeRingsOnly=True, matchValences=False)
        submol = Chem.MolFromSmarts(mcs.smartsString)

        matches_ref = ref.GetSubstructMatches(submol, uniquify=False)
        matches_lig = ligand.GetSubstructMatches(submol, uniquify=False)

        for match_ref in matches_ref:
            for match_lig in matches_lig:
                dist = [
                    (
                        ref.GetConformer().GetAtomPosition(i0)
                        - ligand.GetConformer().GetAtomPosition(ii)
                    ).Length()
                    for i0, ii in zip(match_ref, match_lig)
                ]

                if all([d < distTol for d in dist]):
                    keepMatches = [match_ref, match_lig]
                    return list(zip(*keepMatches))

        raise Exception("ERROR: No MCS found!")

    def _minimize(self, ligand: Mol) -> Mol:
        mcp = deepcopy(ligand)
        ff = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(mcp, confId=0)
        for atidx in [i for i, j in self._mappingRefToLig]:
            ff.UFFAddPositionConstraint(atidx, 0, 200)
        maxIters = 10
        while ff.Minimize(maxIts=4) and maxIters > 0:
            maxIters -= 1

        self._alignToRef(mcp)
        return mcp

    def _alignToRef(self, ligand: Mol) -> None:
        AllChem.AlignMol(
            ligand, self._reference_substructure, atomMap=self._mappingLigToRef
        )
