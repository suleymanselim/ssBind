#!/usr/bin/python
import os, re
from rdkit import Chem
import uuid, shutil, csv
from chem_tools import which

def smina_minimize_score(index, receptor, ligand):

    name = str(uuid.uuid4())
    
    os.makedirs('minimized_conformers', exist_ok=True)
    
    w = Chem.SDWriter(f'{name}.sdf')
    w.write(ligand)
    w.close()

    smina_run = which('smina')

    if smina_run is None:
        smina_run = which('smina.static')
        if smina_run is None:
            raise SystemExit('\nERROR! smina path is not found!!\n')

    RC = os.system(f'''{smina_run} -r {receptor} -l {name}.sdf -o minimized_conformers/{index}_min.sdf --cpu 1 --minimize  > {name}.log 2>&1 ''')
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the smina. See the {} for details.'.format(os.path.abspath("{name}.log\n")))

    RC = os.system(f'''{smina_run} -r {receptor} -l minimized_conformers/{index}_min.sdf --cpu 1 --scoring vinardo --score_only > {name}.log 2>&1 ''')
    if RC != 0:
        raise SystemExit('\nERROR!\nFailed to run the smina. See the {} for details.'.format(os.path.abspath("{name}.log\n")))

    try:
        with open(f'{name}.log', 'r') as file:
            for line in file:
                if line.startswith("Affinity:"):
                    match = re.match(r'Affinity:\s+(-?\d+\.\d+)\s+\(kcal/mol\)', line)
                    if match:
                        affinity_value = match.group(1)
                        if float(affinity_value) > 0:
                            break

                        # Write to CSV file
                        with open('minimized_conformers/Scores.csv', 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([index, affinity_value])
                        break
                    else:
                        print("Invalid format for Affinity in the log.")
                    break
    except FileNotFoundError:
        print(f"Error: File '{name}.log' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Clean up
    for file_to_remove in [f'{name}.log', f'{name}.sdf']:
        os.remove(file_to_remove)

def combine_sdf_files(output_sdf_path, input_sdf_folder, sorted_csv_path):
    
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*') 
    
    sdf_writer = Chem.SDWriter(output_sdf_path)

    files = [f for f in os.listdir(input_sdf_folder) if f.endswith(".sdf")]

    for i in range(len(files)):
        full_path = os.path.join(input_sdf_folder, f'{i}_min.sdf')
        suppl = Chem.SDMolSupplier(full_path)
        for mol in suppl:
            if mol is not None:
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
    
    

