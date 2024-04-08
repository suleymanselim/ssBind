#!/usr/bin/python
import os, re
from rdkit import Chem
import uuid, shutil, csv
from .chem_tools import which
import subprocess
from rdkit import RDLogger

def smina_minimize_score(index, receptor, ligand, out_dir):
    name = str(uuid.uuid4())
    
    ligand_file_path = f'{name}.sdf'
    mol_with_h = Chem.AddHs(ligand, addCoords=True)
    with Chem.SDWriter(ligand_file_path) as w:
        w.write(mol_with_h)

    # Finding smina executable
    smina_executable = which('smina') or which('smina.static')
    if smina_executable is None:
        raise SystemExit('\nERROR! smina path is not found!!\n')


    command = [
        smina_executable, '--receptor', receptor, '--ligand', ligand_file_path,
        '--minimize', '--cpu', '1', '--out', os.path.join(out_dir, f'{index}_min.sdf')
    ]

    log_file_path = f'{name}.log'
    # Running smina 
    with open(log_file_path, 'w') as log_file:
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)

    command = [
        smina_executable, '--receptor', receptor, '--ligand', os.path.join(out_dir, f'{index}_min.sdf'),
        '--score_only', '--cpu', '1', '--scoring', 'vinardo'
    ]

    # Running smina 
    with open(log_file_path, 'w') as log_file:
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)

    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                if 'Affinity' in line:
                    affinity = float(line.split()[1])
                    scores_csv_path = os.path.join(out_dir, 'Scores.csv')
                    with open(scores_csv_path, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([index, affinity])
                    break
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Clean up temporary files
    for file_to_remove in [ligand_file_path, log_file_path]:
        try:
            os.remove(file_to_remove)
        except FileNotFoundError as e:
            print(f"Error: {e}")

def combine_sdf_files(output_sdf_path, input_sdf_folder, sorted_csv_path):
    
    RDLogger.DisableLog('rdApp.*') 
    
    sdf_writer = Chem.SDWriter(output_sdf_path)

    files = [f for f in os.listdir(input_sdf_folder) if f.endswith(".sdf")]

    for i in range(len(files)):
        full_path = os.path.join(input_sdf_folder, f'{i}_min.sdf')
        suppl = Chem.SDMolSupplier(full_path, sanitize=False)
        for mol in suppl:
            sdf_writer.write(mol)

    sdf_writer.close()

    with open(os.path.join(input_sdf_folder, 'Scores.csv'), 'r') as unsorted_csv_file:
        csv_reader = csv.reader(unsorted_csv_file)
        sorted_csv_rows = sorted(csv_reader, key=lambda x: int(x[0]))

    with open(sorted_csv_path, 'w', newline='') as sorted_csv_file:
        csv_writer = csv.writer(sorted_csv_file)
        csv_writer.writerow(['Index', 'Score'])
        csv_writer.writerows(sorted_csv_rows)
    shutil.rmtree(input_sdf_folder)
    
    

