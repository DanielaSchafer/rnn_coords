
import sys
import os
import re

from rdkit import Chem

from normalize_values import get_file, p_end, p_folder, p_types


p = re.compile('\d{2,4}\.\d{3,}')
print(p_end)

def normalize_all_folds(folder_dir, path_to_data):
	for filename in os.listdir(folder_dir):
		print(folder_dir+filename)
		if filename.endswith(".types"):
			write_new_file(folder_dir+filename, path_to_data)
			
def normalize_file(types_file, path_to_data):
	
	lines, energies = get_file(types_file)

	new_line_list = list()
	new_val_list = list()

	for index, line in enumerate(lines):
		sdf_file = extract_sdf_file(line)
		try:
			new_energy = get_normalized_number_by_atom(sdf_file, energies[index], path_to_data)

			new_val_list.append(new_energy)
			new_str = re.sub(p,str(new_energy),str(line),1)
			new_line_list.append(new_str)
		except:
			pass

	return new_line_list

def extract_sdf_file(line):
	p = re.compile('dsgdb9nsd_[\S]{2,}\.sdf')
	sdf_file = p.findall(line)
	return sdf_file

def write_new_file(types_file, path_to_data):
	
	new_lines  = normalize_file(types_file, path_to_data)

	file_name =  p_end.findall(types_file)
	folder_name = (p_folder.split(file_name[0]))
	new_file_name = "/normalized_by_atom_count"+folder_name[1]+"/"+folder_name[2]
	new_folder_name = "/normalized_by_atom_count"+folder_name[1]+"/" 
	
	p_sub = re.compile(file_name[0])
	new_path = re.sub(p_sub, new_file_name, types_file, 1)
	new_folder = re.sub(p_sub, new_folder_name, types_file, 1)
	fold = p_types.split(folder_name[2])[0]

	try: 
		os.mkdir(new_folder)
	except OSError:
		print(new_folder+" already exists")

	with open(new_path, 'w+') as new_fold:
		new_fold.writelines(new_lines)


def get_normalized_number_by_atom(sdf_file, energy, path_to_data):
	atom_count = get_atom_count(sdf_file, path_to_data)	
	norm = float(energy)/float(atom_count)
	return norm

def get_atom_count(sdf_file, path_to_data):
	ms = Chem.SDMolSupplier(path_to_data+sdf_file[0])
	count = list()
	for mol in ms:
		count = mol.GetNumAtoms()
	return count


path = "/net/pulsar/home/koes/dschafer/atomization_energies/fold_files/folds3sdf-t0.8/"
data_dir = "/net/pulsar/home/koes/jss97/datasets/atomization_energies/QM9-sdf/"

normalize_all_folds(path, data_dir)


