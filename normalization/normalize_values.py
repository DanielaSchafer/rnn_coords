#zero mean unit variance
#y = (x-average)/(max-min)

from __future__ import print_function
import sys
import os
import re


#------------------------------------------------------------------------------------
#--------------------------------NORMALIZATION---------------------------------------
#------------------------------------------------------------------------------------

p = re.compile('\d{2,4}\.\d{3,}')
p_normalized = re.compile('-?0\.\d{5,}')
p_end = re.compile('/[\S^/]{1,20}/\S{4,6}\.types')
p_folder = re.compile('/')
p_types = re.compile('\.')

def get_normalized_val(val,min_val, max_val, average):
	new_val = (val-average)/(max_val-min_val)
	return new_val

def get_average_min_max(val_list):
	min_val = val_list[0];
	max_val = 0;
	sum_vals = 0
	for val in val_list:
		sum_vals = sum_vals+val
		if val < min_val:
			min_val = val
		if val > max_val:
			max_val = val

	average = sum_vals/len(val_list)
		
	return 	min_val, max_val, average
	
def normalize_dataset(types_file):
	lines, val_list = get_file(types_file)
	min_val, max_val, average = get_average_min_max(val_list)
	new_val_list = list()
	new_line_list = list()
	for index, val in enumerate(val_list):
		new_val = get_normalized_val(val,min_val, max_val, average)
		new_val_list.append(new_val)
		new_str = re.sub(p,str(new_val),str(lines[index]),1)
		new_line_list.append(new_str)

	return new_line_list, min_val, max_val, average

def write_new_fold(types_file):
	new_lines, min_val, max_val, avg  = normalize_dataset(types_file)

	file_name =  p_end.findall(types_file)
	folder_name = (p_folder.split(file_name[0]))
	new_file_name = "/normalized_"+folder_name[1]+"/"+folder_name[2]
	new_folder_name = "/normalized_"+folder_name[1]+"/" 
	
	p_sub = re.compile(file_name[0])
	new_path = re.sub(p_sub, new_file_name, types_file, 1)
	new_folder = re.sub(p_sub, new_folder_name, types_file, 1)
	fold = p_types.split(folder_name[2])[0]
	print(fold)

	try: 
		os.mkdir(new_folder)
	except OSError:
		print(new_folder+" already exists")

	with open(new_path, 'w+') as new_fold:
		new_fold.writelines(new_lines)
	with open(new_folder+"normalization_info-"+fold+".txt",'w+') as new_info:
		vals = ["avg "+str(avg), "\nmin "+str(min_val), "\nmax "+str(max_val)]	
		new_info.writelines(vals)
	
		

def get_file(types_file):
	data = open(types_file, 'r')
	lines = list()
	energies = list()
	
	for line in data:
		lines.append(line)
		
		templine = p.findall(line)
		energies.append(float(templine[0]))

	return lines, energies


def read_info_folder(folder_path):
	

def get_normalized_file(types_file):
	data = open(types_file, 'r')
	lines = list()
	energies = list()
	
	for line in data:
		lines.append(line)
		
		templine = p_normalized.findall(line)
		print(templine)
		if len(templine) > 0:
			energies.append(float(templine[0]))

	return lines, energies

def normalize_all_folds(folder_path):
	for filename in os.listdir(folder_path):
		print(folder_path+filename)
		if filename.endswith(".types"):
			write_new_fold(folder_path+filename)





#------------------------------------------------------------------------------------
#-------------------------------UNNORMALIZATION--------------------------------------
#------------------------------------------------------------------------------------


#reads out file and returns values from file as a 2D list
def read_result_file(out_file):
	
	#pattern to match correct value
	p_results = re.compile('\d{1}\.\d{1,}')

	data = open(out_file, 'r')
	lines = list() 
	results = list()
	data_list = list()

	for line in data:
		lines.append(line)
		#creates a list of values in line
		results = p_results.findall(line)
		data_list.append(results)
	print(data_list)
	return data_list


#Reads info file and splits values into a dict (ex. info_dict['max'] = 500) 
#returns dict of unnormalization values
def read_info(info_file):
	
	data = open(info_file, 'r')
	info_dict = dict()

	#separates strings in each line and put them in dict	
	for line in data:
		info = line.split()
		info_dict[info[0]] = info[1]

	print(info_dict)
	return (info_dict)
		

#Takes in normalized values, and normalization keys (min, max & avg)
#computes unnormalized value for each element in data_list
#returns list with the unnormalized values of data_list		
def unnormalize_results(data_list, min_val, max_val, avg):
	new_data = list()
	for data in data_list:
		#unnormalization calculation
		new = data*(max_val-min_val)+avg
		new_data.append(new)
		print(new)
	return new_data



#takes in results file and info file
#parses results file into a dict
#converts values in results file with the normalization values from the info file
#creates new file with unnormalized results
def unnormalize_file(out_file, master_info_dicts):

	#returns dict of result values
	out = read_result_file(out_file)

	new_list = list()
	
	#converts each value
	for counter, line in enumerate(out):
		for val in line:
			#find a way to keep track of columns that correspond to each fold
			#unnormalizes value and adds it to list of new values
			new_out = unnormalize_results(line, info_dict["min"], info_dict["max"], info_dict["avg"])
		new_list.append(new_out)
		print(new_out)
	
	#writes new file with unnormalized values
	with open(out_path+"_new", 'w+') as new_fold:
		new_fold.writelines(new_list)


#takes in path of info_files and fold files
#creates and returns a dict of fold_labels[fold_path] = info_dict
def read_info_folder(folder_path):
	
	#need to create regex to match fold and info files
	p_info_file = re.compile()
	p_get_fold_label = re.compile()

	fold_labels = dict()

	files = p_info_file.findall(os.listdir(folder_path))

	for filename in files:
		#gets dict of info vals 
		info = read_info(filename)
		
		#finds corresponding fold file
		f_label = p_get_fold_label.match(filename)

		#adds to dict
		fold_labels[f_label] = info
	return fold_labels


def get_results_files(results_folder):
	#returns list of file paths
	train = "";
	test = "";	
	p_train = re.compile()

	files = os.listdir(results_folder)
	for f in files:
		if p_train.match(f):
			train = f
		else:
			test = f
	
	return train, test
		

#returns a dict of dicts of the normalization values of each fold
def make_master_info_dict(info_files):

	master = dict()	

	for f in info_files:	
		info = read_info(f)
		master[f] = info

	return master
	
#find a way to separate train and test vals	
def master_unnormalize(folder_path, results_folder):
	train, test = get_results_files(results_folder)
	fold_info_files = read_info_folder(folder_path)
	master_info_dict = make_master_info_dict(fold_info_files)

	for results_file in results_files:
		unnormalize_file(results_file)
		
	

		
		

unnormalize_file("/net/pulsar/home/koes/dschafer/atomization_energies/fold_files/normalized_folds3sdf-t0.8/normalization_info-test0.txt", "/net/pulsar/home/koes/dschafer/atomization_energies/scripts/normalized_values/no_rotation_normalized/test")
#read_result_files("/net/pulsar/home/koes/dschafer/atomization_energies/scripts/normalized_values/no_rotation_normalized/test")		

#lines, energies = get_normalized_file("/net/pulsar/home/koes/dschafer/atomization_energies/fold_files/normalized_folds3sdf-t0.8/test0.types")
#unnormalize_results(energies,301.820759419,2602.4373902,1771.11868925)

#normalize_all_folds("/net/pulsar/home/koes/dschafer/atomization_energies/fold_files/folds3sdf-t0.8/")
#write_new_fold("/net/pulsar/home/koes/dschafer/atomization_energies/fold_files/folds3sdf-t0.8/test0.types")

#normalize_dataset("/net/pulsar/home/koes/dschafer/atomization_energies/fold_files/folds3sdf-t0.8/test0.types")
		


