from rdkit.Chem.rdchem import Mol

from ssBind.filter import *
from ssBind.tests.generic import *


def cleanup() -> None:
    """Remove files created upon filtering"""

    for f in [
        "ligand.sdf",
        "filtered.sdf",
    ]:
        try:
            os.remove(f)
        except OSError:
            pass


def test_conformer_filter(receptor_file: str, ligand: Mol) -> None:
    """Test filtering of conformers

    Args:
        receptor_file (str): receptor.pdb protein structure
        ligand (Mol): ligand.mol2 ligand structure
    """

    cleanup()
    ligand_to_sdf(ligand)
    filter = ConformerFilter(receptor_file)
    filter.filter_conformers("ligand.sdf")
    assert files_equal("ligand.sdf", "filtered.sdf")
    cleanup()
