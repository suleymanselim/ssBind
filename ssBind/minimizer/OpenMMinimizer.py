import os

import MDAnalysis as mda
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule, Topology
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
from openmm import LangevinMiddleIntegrator, Platform, app
from openmm.unit import *
from openmmforcefields.generators import (
    GAFFTemplateGenerator,
    SMIRNOFFTemplateGenerator,
)
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


class OpenMMinimizer:

    OUTPUT_PDB = "minimized_conformers.pdb"
    SCORE_FILE = "Scores.csv"

    def __init__(
        self,
        receptor_file: str,
        query_molecule: Mol,
        proteinFF: str = "amber14/protein.ff14SB.xml, amber14/tip3p.xml",
        FF: str = "gaff",
        **kwargs,
    ) -> None:
        """Initialize OpenMM minimization

        Args:
            receptor_file (str): PDB containing protein
            query_molecule (Mol): Ligand as RDKit molecule
            proteinFF (str, optional): Force field to use for protein. Defaults to "sage_ff14sb".
            FF (str, optional): Ligand force field. Defaults to "nagl".
        """
        self._receptor_file = receptor_file
        self._ligand = query_molecule
        self._proteinFF = proteinFF
        self._FF = FF

        platform = kwargs.get("openmm_platform", None)
        if platform is not None:
            self._platform = Platform.getPlatformByName(platform)
        else:
            self._platform = None

        self._integrator = LangevinMiddleIntegrator(
            300 * kelvin, 1 / picosecond, 0.004 * picoseconds
        )

        os.environ["INTERCHANGE_EXPERIMENTAL"] = "1"

    def run_minimization(self, conformers: str = "conformers.sdf") -> None:
        """Run OpenMM minimization

        Args:
            conformers (str, optional): SDF with conformers to minimize.
            Defaults to "conformers.sdf".

        Raises:
            Exception: Unknown FF or proteinFF
        """

        # clean up
        for f in [self.OUTPUT_PDB, self.SCORE_FILE]:
            if os.path.isfile(f):
                os.remove(f)

        # read in topologies with coordinates
        top = Topology.from_pdb(self._receptor_file)
        n_protein = top.n_atoms

        conf = Molecule.from_file(conformers)
        if type(conf) == Molecule:
            conf = [conf]
        ligand = Molecule.from_rdkit(self._ligand)

        # create simulation object based on FFs
        if self._FF == "nagl":
            if self._proteinFF != "sage_ff14sb":
                raise Exception(
                    "NAGL charges are only supported with sage_ff14sb force field"
                )
            openmm_simulation = self._simulation_from_interchange(top, ligand)
        elif self._FF in ["gaff", "smirnoff"]:
            openmm_simulation = self._simulation_openmm(top, ligand)
        else:
            raise Exception(f"Unknown ligand FF: {self._FF}")

        # minimize energy against reference ligand to speed up subsequent minimizations
        openmm_simulation.minimizeEnergy()
        new_positions = openmm_simulation.context.getState(
            getPositions=True
        ).getPositions()
        n_total = len(new_positions)
        protein_pos = new_positions[0:n_protein]

        # minimize conformers
        for i, conformer in enumerate(conf):
            self._minimize_conformer(
                i, conformer, openmm_simulation, protein_pos, n_protein, n_total
            )

        self._minimized_to_sdf()

    def _simulation_openmm(self, top: Topology, ligand: Molecule) -> app.Simulation:
        """Construct OpenMM simulation combining topologies of PDB (top) and ligand
        Molecule, using native openmm.app.Forcefields

        Args:
            top (Topology): Protein topology, read from PDB
            ligand (Molecule): Ligand

        Returns:
            app.Simulation: OpenMM simulation
        """
        top.add_molecule(ligand)
        forcefield = self._get_protein_ff(self._proteinFF)
        if self._FF == "gaff":
            ff = GAFFTemplateGenerator(molecules=ligand)
        else:
            ff = SMIRNOFFTemplateGenerator(molecules=ligand)
        forcefield.registerTemplateGenerator(ff.generator)

        # create system and charge ligand (this takes a few minutes)
        system = forcefield.createSystem(top.to_openmm())

        # create openmm simulation
        openmm_simulation = app.Simulation(top.to_openmm(), system, self._integrator)
        openmm_simulation.context.setPositions(top.get_positions().magnitude)
        return openmm_simulation

    def _simulation_from_interchange(
        self, top: Topology, ligand: Molecule
    ) -> app.Simulation:
        """Create OpenMM simulation from an Interchange

        Args:
            top (Topology): _description_
            ligand (Molecule): _description_

        Returns:
            app.Simulation: _description_
        """

        sage_ff14sb = ForceField(
            "openff-2.2.0.offxml", "ff14sb_off_impropers_0.0.4.offxml"
        )
        protein_intrcg = Interchange.from_smirnoff(
            force_field=sage_ff14sb, topology=top
        )

        NAGLToolkitWrapper().assign_partial_charges(
            ligand, "openff-gnn-am1bcc-0.1.0-rc.3.pt"
        )

        ligand_intrcg = sage_ff14sb.create_interchange(
            ligand.to_topology(), charge_from_molecules=[ligand]
        )

        docked_intrcg = protein_intrcg.combine(ligand_intrcg)

        openmm_simulation = docked_intrcg.to_openmm_simulation(
            self._integrator, platform=self._platform
        )

        return openmm_simulation

    @staticmethod
    def _get_protein_ff(force_fields_to_include: str) -> app.ForceField:
        """Construct protein forcefield using native app.Forcefield

        Args:
            force_fields_to_include (str): Comma-separated string of FFs

        Returns:
            app.ForceField: OpenMM compatible FF
        """
        ff_list = force_fields_to_include.split(",")
        ff_list = [item.strip() for item in ff_list]
        ff = app.ForceField(*ff_list)
        return ff

    @staticmethod
    def _minimize_conformer(
        index: int,
        conformer: Molecule,
        openmm_simulation: app.Simulation,
        protein_pos: quantity.Quantity,
        n_protein: int,
        n_total: int,
    ) -> None:
        """Minimize single conformer and write it to PDB and its potential energy
        to csv.

        Args:
            index (int): Index of this conformer
            conformer (Molecule): Conformer to minimize
            openmm_simulation (app.Simulation): OpenMM simulation
            protein_pos (quantity.Quantity): Positions of proteins to reset
            n_protein (int): Number of protein atoms
            n_total (int): Number of protein+ligand atoms
        """
        new_positions = openmm_simulation.context.getState(
            getPositions=True
        ).getPositions()
        new_positions[0:n_protein] = protein_pos
        new_positions[n_protein:n_total] = (
            conformer.to_topology().get_positions().to_openmm()
        )
        openmm_simulation.context.setPositions(new_positions)
        openmm_simulation.minimizeEnergy()
        state = openmm_simulation.context.getState(getPositions=True, getEnergy=True)
        # write minimized complex
        with open(OpenMMinimizer.OUTPUT_PDB, "a+") as output:
            app.PDBFile.writeModel(
                openmm_simulation.topology,
                state.getPositions(),
                output,
                modelIndex=index,
            )
        # write energy
        potEnergy = state.getPotentialEnergy()._value
        with open(OpenMMinimizer.SCORE_FILE, "a+") as scores_file:
            scores_file.write(f"{index},{potEnergy}\n")

    @staticmethod
    def _minimized_to_sdf() -> None:
        """Extract ligand atoms from PDB to a SDF - this might go away in the future"""
        u = mda.Universe(OpenMMinimizer.OUTPUT_PDB)
        elements = mda.topology.guessers.guess_types(u.atoms.names)
        u.add_TopologyAttr("elements", elements)
        atoms = u.select_atoms("resname UNK")

        sdwriter = Chem.SDWriter("minimized_conformers.sdf")
        for _ in u.trajectory:
            sdwriter.write(atoms.convert_to("RDKIT"))
