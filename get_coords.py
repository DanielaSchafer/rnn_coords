#incorporate atom types with coordinates

import re
import sys
import argparse

parser = argparse.ArgumentParser(description="get coordinates of sdf files")
parser.add_argument('-f','--file',type=str,help="file",required=False)
args = parser.parse_args()

p = re.compile('\d{1}\W{1}\d{4}')
p2 = re.compile('[A-Z]{1,2}')

def assignAtomTypeVals(filename):
	lines = getFile(filename)
	types_dict = dict()
	for counter, line in enumerate(lines):
		types = p2.findall(line)
		for t in types:
			types_dict[t] = counter
	return types_dict
			
		

def getFile(filename):
	f = open(filename,"r")
	lines = list()
	for line in f:
		lines.append(line)
	return lines

	
def getCoords(coord_lines, atom_types):
	coords_arr = list()
	for line in coord_lines:
		atom_type = p2.findall(line)
		coords = p.findall(line)
		coords_num = list()
		for c in coords:
			coords_num.append(float(c))
		if atom_type[0] in atom_types.keys():
			coords_num.append(atom_types[atom_type[0]])
		else:
			coords_num.append(0)
		coords_arr.append(coords_num)
	return coords_arr
		

def getCoordLines(lines):
	coord_lines = list()
	for line in lines:
		l = str(line)
		leading_spaces = len(l) - len(l.lstrip())
		if leading_spaces == 4:
			coord_lines.append(l)
	return coord_lines


def returnCoordList(filename, atom_type_file):
	f = getFile(filename)
	atom_types = assignAtomTypeVals(atom_type_file)
	coord_lines = (getCoordLines(f))
	return (getCoords(coord_lines,atom_types))


