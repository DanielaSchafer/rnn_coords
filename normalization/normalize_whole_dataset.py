from normalize_values import get_file, get_average_min_max, get_normalized_val
import os
import sys
import re


p = re.compile('\d{2,4}\.\d{3,}')


def get_dataset_vals(train, test, path):
	lines_train, val_train = get_file(train)
	lines_test, val_test = get_file(test)
	
	total_lines = lines_train + lines_test
	total_vals = val_train + val_test
	
	min_val, max_val, average = get_average_min_max(total_vals)


	new_folder = path+"normalized/"

	new_types = get_newfiles(path, new_folder)

	types = get_files(path)
	
	try: 
		os.mkdir(new_folder)
	except OSError:
		print(new_folder+" already exists")

	for index, f in enumerate(types):
		new_lines = normalize_file(min_val, max_val, average, f)	
		write_file(new_types[index], new_lines)


	with open(path+"normalization_info.txt",'w+') as new_info:
		vals = ["avg "+str(average), "\nmin "+str(min_val), "\nmax "+str(max_val)]	
		new_info.writelines(vals)


def normalize_file(min_val, max_val, average, types_file):
	lines, val_list = get_file(types_file)
	new_val_list = list()
	new_line_list = list()
	for index, val in enumerate(val_list):
		new_val = get_normalized_val(val,min_val, max_val, average)
		new_val_list.append(new_val)
		new_str = re.sub(p,str(new_val),str(lines[index]),1)
		new_line_list.append(new_str)

	return new_line_list


def write_file(new_path, lines):
	with open(new_path, 'w+') as new_fold:
		new_fold.writelines(lines)

def get_files(path):
	files = list()
	for filename in os.listdir(path):
		print(path+filename)
		if filename.endswith(".types"):
			files.append(path+filename)
	return files

def get_newfiles(path, new_path):
	files = list()
	for filename in os.listdir(path):
		print(new_path+filename)
		if filename.endswith(".types"):
			files.append(new_path+filename)
	return files

path = "/net/pulsar/home/koes/dschafer/atomization_energies/fold_files/folds3sdf-t0.8/"
train = path+"train0.types"
test = path+"test0.types"
get_dataset_vals(train,test,path)

