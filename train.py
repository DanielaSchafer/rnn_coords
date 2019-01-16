from __future__ import unicode_literals, print_function, division
from io import open
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join

from get_coords import returnCoordList
from make_data_dict import getDict
from model import EncoderRNN
from model import DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100
#SOS_token = 0
#EOS_token = 1

def getTensorPairs(path, files_csv, atom_types_file):
	energy_dict = getDict(files_csv)
	tensors = list()
	for file_name, energies in energy_dict.items():
		coords, coords_tensor, energy, energy_tensor = getTensors(path, file_name, energies[0], atom_types_file)
		tensor = (coords, coords_tensor, energy, energy_tensor)
		tensors.append(tensor)
	return tensors
	

def getTensors(path, filename, energy, atom_types_file):
	t3D_energy = list()
	t3D_coords = list()
	energy_list = list()
	energy_list.append(float(energy))
	coords = returnCoordList(path+filename, atom_types_file)

	t3D_energy.append(energy_list)
	t3D_coords.append(coords)

	coords_tensor = torch.tensor(t3D_coords, device=device)
	
	energy_tensor = torch.tensor(t3D_energy, device=device)

	return coords, coords_tensor, energy, energy_tensor

def randomChoice(l):
	return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(pairs):
	tensors = randomChoice(pairs)
	return tensors

tensor_pairs = getTensorPairs("/net/pulsar/home/koes/jss97/datasets/atomization_energies/QM9-sdf/", "sdfData_short.csv","atom_types")

criterion = nn.MSELoss()

n_input = 64
n_hidden = 128 

import time
import math

n_iters = 10
print_every = 5
plot_every = 1

current_loss = 0
all_losses = []

def timeSince(since):
	now = time.time()
	s = now -since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

start = time.time()


pairs = (getTensorPairs("/net/pulsar/home/koes/jss97/datasets/atomization_energies/QM9-sdf/", "sdfData_short.csv","atom_types"))
#print(pairs)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

	encoder_hidden = encoder.hidden_size

	print(encoder_optimizer, decoder_optimizer)
	
	
	decoder_optimizer.zero_grad()
	encoder_optimizer.zero_grad()
	
	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)
	print(input_length)



	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	loss = 0

	encoder_output, encoder_hidden = encoder(input_tensor)

 	decoder_input = encoder_output
	encoder_hidden = encoder_hidden.squeeze(1)
	print("hello")
	print(encoder_hidden)
	decoder_hidden = encoder_hidden

	decoder_output = decoder(decoder_input, decoder_hidden)


	l1 = list()
	l1.append(float(decoder_output[0][0][0]*10000))
	l2 = list()
	l2.append(l1)
	out_tensor = torch.tensor(l1)
	print()
	out_val = list(decoder_output)


	decoder_out_val = out_tensor.item()
	target_val = target_tensor.item()
	target_tensor_1d = torch.tensor(target_val)
#	print(decoder_out_val, target_val)

	#output = (torch.randn(decoder_output_val).float())
	#target = (torch.FloatTensor(10).uniform_(0, 120).long())

#	print(target_tensor_1d, out_tensor)

	#loss = criterion(out_tensor, target_tensor_1d)#out_tensor, target_tensor)
	#***********************#
	#HEY DANIELA THIS IS JOCELYN 
	#THE TARGET_TENSOR_1D IS A SINGLE VALUE, SO I NEEDED TO GET A SINGLE VALUE OUTPUT
	#FROM THE DECODER TO GIVE THE LOSS AS INPUT; WHEN YOU GET RID OF THE DECODER YOU SHOULD
	#JUST GET A SINGLE VALUE OUTPUT IN REALITY AND SO THIS WILL NOT BE NECESSARY
	#***********************#
	loss = criterion(decoder_output[0][-1][-1], target_tensor_1d)#out_tensor, target_tensor)

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length


def tensorsFromPair(pair):
	out =  (pair[1],pair[3])	
#	print(out)
	return(out)



def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
	start = time.time()
	plot_losses = []
	print_loss_total = 0
	plot_loss_total = 0

	print(list(encoder.parameters()))
	print(list(decoder.parameters()))

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

	training_pairs = [tensorsFromPair(random.choice(pairs))
		for i in range(n_iters)]

	criterion = nn.MSELoss()

	for iter in range(1, n_iters+1):
		training_pair = training_pairs[iter - 1]
		input_tensor = training_pair[0]
		target_tensor = training_pair[1]

		loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
		print_loss_total += loss
		plot_loss_total += loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0

	showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
	plt.figure()
	fig, ax = plt.subplots()

	loc = ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plot.plot(points)


def evalutate(encoder, decoder, example, max_length=MAX_LENGTH):
	with torch.no_nograd():
		input_tensor = tensorFromSentence(input_lang, sentence)
		input_length = input_tensor.size()[0]
		encoder_hidden = encoder.initHidden()

		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

		for ei in range(input_length):
			encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
			encoder_outputs[ei] += encoder_output[0,0]

		decoder_input = torch.tensor([[SOS_token]], device=device)

		decoder_hidden = encoder_hidden

		decoded_val = []
		decoder_attentions = torch.zeros(max_length, max_length)

		for di in range(max_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(
				decoder_input, decoder_hidden, encoder_outputs)

			decoder_attentions[di] = decoder_attention.data
			topv, topi = decoder_output.data.topk(1)
			if topi.item() == EOS_token:
				decoded_val.append('<EOS>')
				break
			else:
				decoded_val.append(output_lang.index2word[topi.item()])

			decoder_input = topi.squeeze().detach()

		return decoded_h, decoder_attentions[:di + 1]


hidden_size = 256
input_size = 4
output_size = 1
encoder1 = EncoderRNN(input_size, hidden_size).to(device)
decoder1 = DecoderRNN(input_size, hidden_size).to(device)

trainIters(encoder1, decoder1, 75000, print_every=5000)

#def evaluateRandomly(encoder, decoder, n=10):
#	for i in range(n):
#		pair = random.choice(pairs)
#		print('>', pair[0])
#		print('=', pair[1])
#		output_val, attentions = evaluate(encoder, decoder, pair[0])
#		output_sentence = ' '.join(output
			
#	coords_tensor = pair[1]
#	energy = pair[2] 
#	energy_tensor = pair[3]
#	output, loss = train(energy_tensor, coords_tensor)
#	current_loss += loss
#
#	if iter % print_every == 0:
#		guess = utput
#		correct = 'y' if guess == energy else 'x (%s)' % energy
#		print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter /n_iters / n_iters * 100, timeSince(start), loss, line, guess, correct))
#
#	if iter % plot_every == 0:
#		all_losses.append(current_loss / plot_every)
#		current_loss = 0

#import mathplotlib.pyplot as plt
#import mathplotlib.ticker as ticker
#
#plt.figure()
#plt.plot(all_losses)
#
	

#getTensor("./dsgdb9nsd_119016.sdf")

#print(getTensorPairs("/net/pulsar/home/koes/jss97/datasets/atomization_energies/QM9-sdf/", "sdfData_short.csv","atom_types"))
