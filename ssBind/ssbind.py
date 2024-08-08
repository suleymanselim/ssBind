import multiprocessing as mp
import os
import uuid

from ssBind.filter import ConformerFilter
from ssBind.generator import *
from ssBind.minimizer import *
from ssBind.posepicker import PosePicker

# Substructure-based alternative BINDing modes generator for protein-ligand systems


class SSBIND:

    def __init__(self, **kwargs):

        kwargs.setdefault("nprocs", mp.cpu_count())
        kwargs.setdefault("numconf", 1000)
        kwargs.setdefault("curdir", os.getcwd())
        kwargs.setdefault(
            "working_dir", os.path.join(kwargs["curdir"], str(uuid.uuid4()))
        )
        self._kwargs = kwargs
        self._generator = None
        self._filter = None
        self._minimizer = None
        self._posepicker = None

    def generate_conformers(
        self,
    ):
        """
        Generates conformers using RDKit or dihedral angle sampling.

        """

        generator_type = self._kwargs.get("generator")

        if generator_type == "rdkit":
            self._generator = RDkitConformerGenerator(
                **self._kwargs,
            )
        elif generator_type == "angle":
            self._generator = AngleConformerGenerator(
                **self._kwargs,
            )
        elif generator_type == "rdock":
            self._generator = RdockConformerGenerator(
                **self._kwargs,
            )
        elif generator_type == "plants":
            self._generator = PlantsConformerGenerator(**self._kwargs)
        else:
            raise Exception(f"Invalid conformer generator: {generator_type}")

        self._generator.generate_conformers()

    def filter_conformers(self):
        """Filter conformers which have clashes etc"""

        generator_type = self._kwargs.get("generator")

        if generator_type in ["rdkit", "angle"]:
            self._filter = ConformerFilter(**self._kwargs)
            self._filter.filter_conformers()

    def run_minimization(
        self,
        conformers="conformers.sdf",
    ):
        """
        Performs minimization and scoring.

        """
        minimizer_type = self._kwargs.get("minimize")

        if minimizer_type is None:
            pass
        elif minimizer_type == "gromacs":
            self._minimizer = GromacsMinimizer(**self._kwargs)
        elif minimizer_type == "smina":
            self._minimizer = SminaMinimizer(**self._kwargs)
        else:
            raise Exception(f"Invalid minimizer: {minimizer_type}")

        self._minimizer.run_minimization(conformers)

    def clustering(
        self,
        conformers: str = "conformers.sdf",
        scores: str = "Scores.csv",
    ):
        """
        Performs clustering based on the conformational distance (RMSD) matrix.

        """
        self._posepicker = PosePicker(**self._kwargs)
        self._posepicker.pick_poses(conformers, scores)
