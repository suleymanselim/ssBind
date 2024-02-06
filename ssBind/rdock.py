#!/usr/bin/python

from chem_tools import MolFromInput
import os


def get_tethered(refmol, ligand, out): ###https://github.com/Discngine/rdkit_tethered_minimization/blob/master/tetheredMinimization.py

    from rdkit import Chem
    from rdkit.Chem import rdFMCS, AllChem 
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
	
    ligandHs = Chem.AddHs(ligand, addCoords=True)
	
    mcs = rdFMCS.FindMCS([refmol, ligandHs], completeRingsOnly=True)

    print(mcs.smartsString)
    refmolcore = Chem.AllChem.ReplaceSidechains(refmol,Chem.MolFromSmarts(mcs.smartsString, mergeHs=True))

    core=AllChem.DeleteSubstructs(refmolcore, Chem.MolFromSmiles('*'))
    core.UpdatePropertyCache()
            
    GetFF=lambda x,confId=-1:AllChem.MMFFGetMoleculeForceField(x,AllChem.MMFFGetMoleculeProperties(x),confId=confId)
    AllChem.ConstrainedEmbed(ligandHs,core, getForceField=GetFF, useTethers=True)
    tethered_atom_ids=ligandHs.GetSubstructMatches(Chem.MolFromSmarts(mcs.smartsString, mergeHs=True))[0]

    atoms = map(lambda x:x+1, list(tethered_atom_ids))
    atoms_string = ','.join(str(el) for el in atoms)

    w=Chem.SDWriter(f'{out}.sdf')
    w.write(ligandHs)
    w.flush()
    
    ligandHs.SetProp('TETHERED ATOMS',atoms_string)
        
    w=Chem.SDWriter(f'{out}.sd')
    w.write(ligandHs)
    w.flush()

    return out


def prepare_receptor(RECEPTOR_FILE = 'receptor.pdb', RECEPTOR_FLEX: float = 3.0, REF_MOL = 'ligand.sdf', 
	SITE_MAPPER: str = 'RbtLigandSiteMapper', RADIUS: float = 6.0, SMALL_SPHERE: float = 1.0, 
	MIN_VOLUME: int = 100, MAX_CAVITIES: int = 1, VOL_INCR: float = 0.0, 
	GRIDSTEP: float = 0.5, SCORING_FUNCTION: str = 'RbtCavityGridSF', WEIGHT: float = 1.0):
    """
    rbcavity – Cavity mapping and preparation of docking site 

    For a detailed description of the parameters please see rDock documentation https://rdock.github.io/documentation/html_docs/reference-guide/cavity-mapping.html
        
    """

    params = """RBT_PARAMETER_FILE_V1.00
TITLE rDock

RECEPTOR_FILE {}
RECEPTOR_FLEX {}

##################################################################
### CAVITY DEFINITION: REFERENCE LIGAND METHOD
##################################################################
SECTION MAPPER
	SITE_MAPPER {}
	REF_MOL {}
	RADIUS {}
	SMALL_SPHERE {}
	MIN_VOLUME {}
	MAX_CAVITIES {}
	VOL_INCR {}
	GRIDSTEP {}
END_SECTION

#################################
#CAVITY RESTRAINT PENALTY
#################################
SECTION CAVITY
	SCORING_FUNCTION {}
	WEIGHT {}
END_SECTION
            
""".format(RECEPTOR_FILE, RECEPTOR_FLEX, SITE_MAPPER,
        REF_MOL, RADIUS, SMALL_SPHERE, MIN_VOLUME, MAX_CAVITIES, 
        VOL_INCR, GRIDSTEP, SCORING_FUNCTION, WEIGHT)
        

    with open('rbdock.prm', 'w') as f:
        f.write(params)

    RC = os.system('''{} -was -d -r rbdock.prm > rbcavity.log 2>&1 '''.format(which('rbcavity')))
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the rbcavity. see the %s for details.'%os.path.abspath("rbcavity.log\n"))

    
def run_rdock(ligand, output, nruns=10, TRANS_MODE='TETHERED', ROT_MODE='TETHERED', DIHEDRAL_MODE='FREE', MAX_TRANS=1.0, MAX_ROT=30.0):
    """
    rbdock – the rDock docking engine itself.
        
    For a detailed description of the tethered parameters please see rDock documentation https://rdock.github.io/documentation/html_docs/reference-guide/cavity-mapping.html
    
    """

    params="""
################################
# TETHERED SCAFFOLF
################################
SECTION LIGAND
	TRANS_MODE {}
	ROT_MODE {}
	DIHEDRAL_MODE {}
	MAX_TRANS {}
	MAX_ROT {}
END_SECTION
""".format(TRANS_MODE,ROT_MODE, DIHEDRAL_MODE, MAX_TRANS,MAX_ROT)
            


    with open('rbdock.prm') as f:
        rbcavity_params = f.readlines()
    with open('rbdock.prm', 'w') as rbdock_params:
        rbdock_params.writelines(rbcavity_params)
        rbdock_params.write(params)
        

    RC = os.system('''{} -i {} -o {} -r rbdock.prm -p dock.prm -n {} > rbdock.log 2>&1 '''.format(which('rbdock'), ligand, output, nruns))
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the rDock. see the %s for details.'%os.path.abspath("rbdock.log\n"))

    return
