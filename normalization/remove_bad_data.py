
import sys
import os
import re

from rdkit import Chem

from normalize_values import get_file, p_end, p_folder, p_types, p
from normalize_by_atoms import get_normalized_number


def remove_bad_data(types_file, path_to_data):
	
	lines, energies = get_file(types_file)

	new_line_list = list()
	new_val_list = list()

	for index, line in enumerate(lines):
		sdf_file = extract_sdf_file(line)
		try:
			new_energy = get_normalized_number(sdf_file, energies[index], path_to_data)

			new_val_list.append(energies[index])
			new_str = re.sub(p,str(energies[index]),str(line),1)
			new_line_list.append(new_str)
		except:
			pass

	return new_line_list


def write_new_file(types_file, path_to_data):
	
	new_lines  = remove_bad_data(types_file, path_to_data)

	file_name =  p_end.findall(types_file)
	folder_name = (p_folder.split(file_name[0]))
	new_file_name = "/remove_bad_data"+folder_name[1]+"/"+folder_name[2]
	new_folder_name = "/remove_bad_data"+folder_name[1]+"/" 
	
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




def edit_all_folds(folder_dir, path_to_data):
	for filename in os.listdir(folder_dir):
		print(folder_dir+filename)
		if filename.endswith(".types"):
			write_new_file(folder_dir+filename, path_to_data)


path = "/net/pulsar/home/koes/dschafer/atomization_energies/fold_files/folds3sdf-t0.8/"
data_dir = "/net/pulsar/home/koes/jss97/datasets/atomization_energies/QM9-sdf/"
edit_all_folds(path, data_dir)



normalize_all_folds(path, data_dir)


