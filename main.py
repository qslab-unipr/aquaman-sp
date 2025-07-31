import qsp_srbb_circuit as USC
import config
import importlib
import inspect
import pennylane as qml
from pennylane import numpy as np
import argparse
import autograd.numpy as anp
import cmath
import sympy as sp
import pickle
import scipy.optimize
import time
from scipy.stats import unitary_group
from qiskit.quantum_info.analysis import hellinger_fidelity 
import warnings
import os
import matplotlib.pyplot as plt
import eval
warnings.filterwarnings("ignore", category=anp.ComplexWarning)

def get_args():
	parser = argparse.ArgumentParser()	 

	parser.add_argument('--run_name', type=str, default="run_1", help='name of current run')

	parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
	parser.add_argument('--batch_size', type=int, default=64, help='number of elements in batch')

	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--opt', type=str, default='Adam', choices=['Adam', 'Nelder_Mead'], help = 'optimizer used for training')

	parser.add_argument('--loss', type=str, default = 'trace_distance', choices=['trace_distance', 'frob', 'fidelity'], help='loss function')
	parser.add_argument('--num_samples', type=int, default=1000, help="if trace or fidelity losses are used, you can set number of states for training set")
	parser.add_argument('--test_size', type=int, default=500, help="size of testset")

	parser.add_argument('--num_layer', type=int, default = 1, choices=range(1, 10), help = 'number of circuit repetition')
	parser.add_argument('--only_test', type=bool, default=False, help='True if you dont want to train the network, but retrieve the data from file')

	parser.add_argument('--device', type = str, default = "sim", choices = ["sim", "hw", "iqm"], help = 'choose if u want to execute on the pennylane simulator or on ibm real hw')
	parser.add_argument('--n_qubits', type = int, default = 2, help = 'number of qubits')

	parser.add_argument('--path', type=str, default=os.path.join(os.getcwd(), "results/"), help="where to save/load the files")

	parser.add_argument('--state', type=bool, default=False, choices=[True, False],
					 help = "if True, the VQC will achieve the desired state, otherwise it will achieve the probabilities associated to it")

	return parser.parse_args()




def main(args):

	dev = config.set_device(args.n_qubits, args.device) #dynamic initialization of the device
	print("Device:", dev)
	print("Device type:", type(dev))

	ideal_module_name=f'ideal_circuitN{args.n_qubits}'
	ideal_module=importlib.import_module(ideal_module_name)#takes the corresponding ideal circuit
	
	#execute program to generate a random state,
	#a specific state can be also set, as shown below.
	for i in range(1):
		eval.eval("random", ideal_module, args, i, dev)
	"""
	if args.n_qubits == 2:
		main_program('bell', ideal_module, args, 0)
		main_program('hi', ideal_module, args, 0)
		main_program('hh', ideal_module, args, 0)
		main_program('deutsch_balanced', ideal_module, args, 0)
		main_program('deutsch_unbalanced', ideal_module, args, 0)
	elif args.n_qubits == 3:
		main_program('ghz', ideal_module, args, 0)
		main_program('hhh', ideal_module, args, 0)
		main_program('cnoth', ideal_module, args, 0)
		main_program('deutsch_jozsa_balanced', ideal_module, args, 0)
		main_program('hxx_iyz', ideal_module, args, 0)
	elif args.n_qubits == 4:
		main_program('hhhh', ideal_module, args, 0)
		main_program('deutsch_jozsa_balanced', ideal_module, args, 0)
		main_program('deutsch_jozsa_unbalanced', ideal_module, args, 0)
	"""
	
		
	

if __name__ == "__main__":
	args = get_args()
	print(args)
	main(args)
	
