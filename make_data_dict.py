import argparse
import sys
import re
from get_coords import getFile

parser = argparse.ArgumentParser(description="get atomization energies from csv")
parser.add_argument('-f','--file',type=str,help="file",required=False)
args = parser.parse_args()

p = re.compile('[-]{0,1}\d{1,}\.\d{1,}')

def getDict(filename):
	energies = dict()
	lines =	getFile(filename)
	for line in lines:
		str_line = str(line)
		vals = str_line.split(",")
		energy = list()
		for counter, v in enumerate(vals):
			num = p.findall(v)
			if counter != 0 and len(num) > 0:
				energy.append(float(num[0]))
		
		energies[vals[0]]= energy
	return energies

#print(getDict(args.file))

		
