import numpy as np
import os
import sys
import re

from normalize_by_atoms import extract_sdf_file, get_atom_count, p_end, p_folder

output = ""
fold_folder = ""

#the main method thing
def runner(output_path, data_path, fold_folder):

	#get list of types file paths
	types_files = get_types_files(fold_folder)

	#get list of normalized results files
	rmsd_filenames_norm = get_rmsd_file_names(output_path)

	for n in rmsd_filenames_norm:
		#get the correct fold file		
		output_path, fold_path = get_corresponding_types_file(n, types_files)

		sdf_paths = get_sdf_list(fold_path)

		#get list of normalized exp values and normalized target values
		exp_vals, target_vals = open_rmsd_file(n)


		unnormalized_rmsd = unnormalize_results(exp_vals, target_vals, sdf_paths, data_path)

		new_filename = get_new_filename(output)		
		write_data(new_filename, unnormalized_rmsd[index])	

def get_new_filename(filename_old):
	return filename_old+"_unnormalized"

def get_new_folder(folder_old)
	folder_old.strip()


#returns files that end in '.types' from a given folder
def get_types_files(folder_dir):
	types_files = list()
	for filename in os.listdir(folder_dir):
		print(folder_dir+filename)
		if filename.endswith(".types"):
			types_file.append(folder_dir+filename)
	return types_files

#writes data into a folder given new lines and a new file path
def write_data(file_name, new_lines):
	with open(file_name, 'w+') as f:
		f.writelines(new_lines)	

def get_sdf_list(fold_path, path_to_sdf):
	lines = open(fold_path, 'r')	
	sdf_files = list()
	for line in lines:
		sdf_files.append(extract_sdf_file(line))
	return sdf_files

#takes in sdf path, exp normalized energy, path to types, path to sdf files
def get_unnormalized_number_by_atom(sdf_file, n_energy, path_to_data):
	atom_count = get_atom_count(sdf_file, path_to_data)
	un_norm = float(n_energy)*float(atom_count)
	return un_norm

#makes new folder if it does not already exist
def new_folder(folder):
	try: 
		os.mkdir(new_folder)
	except OSError:
		print(new_folder+" already exists")

def get_new_folder_and_file_name(type_name, types_file):
	
	file_name =  p_end.findall(types_file)
	folder_name = (p_folder.split(file_name[0]))
	new_file_name = "/normalized_by_atom_count"+folder_name[1]+"/"+folder_name[2]
	new_folder_name = "/normalized_by_atom_count"+folder_name[1]+"/" 
	
	p_sub = re.compile(file_name[0])
	new_path = re.sub(p_sub, new_file_name, types_file, 1)
	new_folder = re.sub(p_sub, new_folder_name, types_file, 1)
	fold = p_types.split(folder_name[2])[0]

	return new_path, new_folder

def get_normalized_output(path):
	data = list()

	return data


#links output file to the correct types fold file
def get_corresponding_types_file(output_file, fold_list):
	types_file = ""

	return output_file, types_file 


#returns exp and real values from output data
def open_rmsd_file(path):
	
	data = open(types_file, 'r')
	exp = list()
	real = list()
	
	print(types_file)
	
	for line in data:
		parts = line.split()
		exp.append(parts[0])
		real.append(parts[1])

		print(parts)

	return exp, real


#returns paths of output files 
#experimental and actual data
def get_rmsd_file_names(path):

	train = list()
	test = list()

	for filename in os.listdir(folder_path):
		print(folder_path+filename)
		if filename.endswith(".finaltest"):
			test.append(path+filename)
		elif filename.endswith(".finaltrain"):
			train.append(path+filename)	
	return train, test

#unnormalizes an output file given a list of outputs, a corresponding list of fold data, list of sdf files from fold, and a path to sdf files
#returns list of unnormalized rmsd values
def unnormalize_results(output_list, target_list, sdf_list, path_to_data):
	new_output = list()
	new_rmsd = list()
	for index, out in enumerate(output_list):
		un_exp = get_unnormalized_number_by_atom(sdf_list[index], out, path_to_data)
		un_target = get_unnormalized_number_by_atom(sdf_list[index], target_list[index], path_to_data)
		new_output.append(unn)
		new_rmsd.append(rmse(un_exp, un_target))
	return new_output
	
		
#gets the rmse of a prediction and target value
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())



