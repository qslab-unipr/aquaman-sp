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
import tree
import utils
import losses
import dataset
warnings.filterwarnings("ignore", category=anp.ComplexWarning)

def eval(c, ideal_module, args, index, device):
	"""
	Main code. Execution of training and testing

	Args:
		c (str): Name of the ideal circuit to approximate
		ideal_module (): Name of the module where the circuit c is in.
		args (): Given arguments.
		index (int): Number of state
		device (): Selected Device
	"""

	############# PRELIMINAR OPERATION ###############
	#define directory location
	dir = args.path + str(args.n_qubits) + 'qubit/' + args.run_name + '_N' + str(args.n_qubits) + args.loss + args.opt + str(args.epochs)
	pathName = 'state' + str(index) + args.run_name + c + 'LR' + str(args.lr) + args.opt + args.loss + str(args.epochs) + str(args.num_samples) + "BS" + str(args.batch_size)
	
	if not os.path.exists(args.path + str(args.n_qubits) + 'qubit/'):
		os.makedirs(args.path + str(args.n_qubits) + 'qubit/')

	if c == 'random': 
		#generate state to be approximated
		real_part=np.random.normal(size=2**args.n_qubits)
		imag_part=np.random.normal(size=2**args.n_qubits)
		random_complex_vector=real_part+1j*imag_part
		norm=np.linalg.norm(random_complex_vector)

		state=random_complex_vector/norm
	else:
		#obtain the state from a predefined circuit defined in ideal_circuiN* modules
		U=np.zeros(shape=(2**args.n_qubits,2**args.n_qubits),dtype=complex)
		circuit=getattr(ideal_module,c)
		state = circuit(device, args.n_qubits)([], U, args.device, True)
	
		#print(state)
		#input()
	print(f'Generated State: {state}\n')

	if not args.only_test:
		fileParams = open(args.path + 'ideal_' + pathName + '.obj', 'wb')
						
		pickle.dump(state, fileParams)
		fileParams.close()
	
	######## Unitary Matrix Computation ########
	
	#generate the tree where in each root we will insert the corresponding amplitude
	tree_root = tree.build_sum_tree(state)
	#print_tree(tree_root)
	#input()
	
	#compute theta of multiplexors in order to calculate the U needed to prepare the desired state
	num_theta_multiplexor = 1
	for n in range(1, args.n_qubits):
		num_theta_multiplexor += 2**n
	theta_multiplexor = np.empty(shape=num_theta_multiplexor)

	#retrieve the list of amplitudes
	amplitude_dic = tree.explore_tree(tree_root)
	#print(amplitude_dic)
	#input()

	#compute the first theta
	theta_multiplexor[0] = np.arccos(amplitude_dic.get("0") / amplitude_dic.get(""))

	#computation of other thetas for multiplexors
	count_bit = 0
	i=1
	while(i < num_theta_multiplexor):
		count_bit += 1
		for j in range(i, i + 2**(count_bit)):
			#each theta is computed as the arccos of the node amplitude divided by its father amplitude
			amp1_key = f"{bin(j-i)[2:].zfill(count_bit)}0"
			amp2_key = f"{bin(j-i)[2:].zfill(count_bit)}"
			if amplitude_dic.get(amp2_key) != 0:
				theta_multiplexor[j] = np.arccos(amplitude_dic.get(amp1_key) / amplitude_dic.get(amp2_key))
			else:
				theta_multiplexor[j] = 0

		i += 2**(count_bit)
	#print(theta_multiplexor)
	#input()

	#compute U matrix
	U = []
	
	#the U matrix is computed as a multiplexor of Ry rotations.
	#we will start from the last qubit and multiply between them all the matrices with the different controls value,
	# until we will reach the first which has only an Ry rotation without control qubit
	theta_iterator = num_theta_multiplexor-1
	for n in range(args.n_qubits, 1, -1):
		temp = qml.matrix(qml.ControlledQubitUnitary(qml.RY.compute_matrix(theta_multiplexor[theta_iterator]*2), wires = n, control_wires = range(0, n-1), control_values = [1 for _ in range(0, n-1)]))
		#print(temp)
		#input()
		if n != args.n_qubits:
			temp = np.kron(temp, qml.Identity.compute_matrix(n_wires = args.n_qubits - n))
		
		theta_iterator -= 1

		for control in range(2**(n-1)-2, -1, -1):
			
			c_values = f"{bin(control)[2:].zfill(n-1)}"
			#print(c_values)
			#input()
			
			#print(c_values)
			temp2 = qml.matrix(qml.ControlledQubitUnitary(qml.RY.compute_matrix(theta_multiplexor[theta_iterator]*2), wires = n, control_wires = range(0, n-1), control_values = [int(c) for c in c_values]))
			if n != args.n_qubits:
				temp2 = np.kron(temp2, qml.Identity.compute_matrix(n_wires = args.n_qubits - n))
			temp = temp @ temp2

			theta_iterator -= 1
		
		if len(U) == 0:
			U = temp
		else:
			U = U @ temp
		
	#multiply by the Ry rotaion on the first qubit
	U = U @ np.kron(qml.RY.compute_matrix(theta_multiplexor[0]*2), qml.Identity.compute_matrix(n_wires = args.n_qubits-1))
	polar_state = []
	for s in state:
		polar_state.append(cmath.polar(s))

	
	#if we want to generate the precise state and not just its probabilities, we need the U that generate its phase
	if args.state:
		#generate unitary matrix
		U_phase=np.zeros(shape=(2**args.n_qubits,2**args.n_qubits),dtype=complex)

		for i in range(2**args.n_qubits):
			U_phase[i, i] =   np.exp(1j * polar_state[i][1])

	#print(U@((np.conj(U)).T))

	########### GENERATE TRAINING SET ###################

	#create the dataset of training and test sets
	X_train,X_test=dataset.create_dataset(args)

	#Used to run the circuit indicated by the c variable
	circuit=getattr(ideal_module,"random")
	
	#return the ideal output state generated by the circuit (random or predefined) for each training sample 
	y_ideal=[circuit(device, args.n_qubits)(x,U, args.device) for x in X_train]


	########## COMPUTE SU MATRICES #########

	#if we want the precise state, we can have 2^n_qubits different SU based on the phases and we have to choose the correct one
	if args.state:
		y_ideal_phase = [circuit(device, args.n_qubits)(x, U_phase, args.device, True) for x in X_train]

		#calculate the determinant of the unitary that we want to approximate
		det=np.linalg.det(U_phase)

		#calculate the 2^n roots of the determinant that give the 2^n possible SU's
		z=sp.symbols('z', complex=True)
		equation=z**(2**args.n_qubits)-det 
		solutions=sp.solve(equation,z)
		#print(solutions)
		solutions=[complex(sol.evalf()) for sol in solutions]
		#print("roots of the determinant:", solutions)
			
		#sorting the four roots: ++ -> -+ -> -- -> +- (counterclockwise arrangement starting from 0 radiant in the unitary cfr)
		polar=[cmath.polar(s) for s in solutions]
		#print(polar)
		polar=[p if p[1] >= 0 else (p[0], np.pi + (np.pi + p[1])) for p in polar]
		polar=utils.bubble_sort(polar)
		#print(polar)
		solutions = [cmath.rect(r,theta) for (r,theta) in polar]
		#print(solutions)

		#calculate the SU's associated to the 2^n roots
		listSU=[]

		for i in range(len(solutions)):
			SU=U_phase/solutions[i]	
			#format_matrix(SU)
			listSU.append(SU)
			#print(np.linalg.det(SU))
		#input()
		
		#save all the ideal SU
		"""
		with open(dir + 'SU_Results' + '.txt', "a") as file:
			file.write(c)
			file.write("\nIDEAL SU:\n")
			for SU in listSU:
				file.write(format_matrix2(SU))
				file.write("\n\n")
		"""
	x_max=2**(args.n_qubits-1)-1 #number of ProdT_factors and M_factors	

	if args.only_test==False:

		######################## TRAINING OF THE NETWORK #########################

		#execute the training
		start_time=time.time() #count time taken by the training 
		#the ideal U is given as argument because the vidal circuit for two qubit is used for unitary matrices and not for SU
		print("MODULO TRAINING")
		params, _, loss, norm, U_approx, SU_ideal, rot_count = training(X_train, y_ideal, args.batch_size, args.lr, args.epochs, args.loss, args.opt,
																args.num_layer, args.n_qubits, solutions, listSU, listSU[0], dir, x_max, U, device, c, pathName, args.path,
																False, args)
		if args.state:
			print("PHASE TRAINING")
			params_phase, U_approx_phase, loss, norm, SU_approx_phase, SU_ideal, rot_count_phase = training(X_train, y_ideal_phase, args.batch_size, args.lr, args.epochs, args.loss, args.opt,
																args.num_layer, args.n_qubits, solutions, listSU, listSU[0], dir, x_max, U_phase, device, c, pathName, args.path,
																True, args)
		end_time=time.time()
		elapsed_time=end_time-start_time
	else:
		
		####### Retrieve all the data if we don't want to train #######
		#retrieve the state
		state_app = open(args.path + 'ideal_' + pathName + '.obj', 'rb')
		state = pickle.load(state_app)
		state_app.close()

		#retrieve the U matrix which approximate the modulo
		fileu_app = open(args.path + 'U_appModulo' + pathName + '.obj', 'rb')
					
		U_approx = pickle.load(fileu_app)
		fileu_app.close()

		#retrieve the U matrix which approximate the the phase
		fileu_app = open(args.path + 'U_appPhase' + pathName + '.obj', 'rb')
					
		U_approx_phase = pickle.load(fileu_app)
		fileu_app.close()

		#retrieve the number of rotation for modulo
		rot_count_app = open(args.path + 'rot_countModulo' + pathName + '.obj', 'rb')
		rot_count = pickle.load(rot_count_app)
		rot_count_app.close()

		#retrieve the number of rotation for phase
		rot_count_app = open(args.path + 'rot_countPhase' + pathName + '.obj', 'rb')
		rot_count_phase = pickle.load(rot_count_app)
		rot_count_app.close()

		#retireve theta angles for modulo
		params_app = open(args.path + 'paramsModulo' + pathName + '.obj', 'rb')
		params = pickle.load(params_app)
		params_app.close()

		#retireve theta angles for phase
		params_app = open(args.path + 'paramsPhase' + pathName + '.obj', 'rb')
		params_phase = pickle.load(params_app)
		params_app.close()
	

	
	ideal_probs = []
	print("Ideal Probabilities: ")
	for s in state:
		print(s.real**2 + s.imag**2, end=" ")
		ideal_probs.append(s.real**2 + s.imag**2)
	
	print("\nApproximated probabilities: ")
	print(USC.test_srbb_method(device, args.n_qubits)(U_approx, False))
	print("Hellinger Distance between ideal and approximated probabilities: ")
	print(losses.hellinger(ideal_probs, USC.test_srbb_method(device, args.n_qubits)(U_approx, False)))

	if args.state:
		#compute ideal density matrix
		dm = USC.amplitude_density(device, args.n_qubits)(state)
		print("\n\n\n\nIdeal State: ")
		print(state)
		print("Approximated state: ")
		print(USC.test_srbb_method(device, args.n_qubits)(U_approx_phase @ U_approx, True))
		#print(USC.test_srbb_method(U_approx_phase @ U_approx, False))
		print("Trace distance between ideal and approximated state: ")
		print(qml.math.trace_distance(dm, USC.test_srbb_method(device, args.n_qubits)(U_approx_phase @ U_approx, True, True)))

	#input()
	with open(dir + '_results' + '.txt', 'a') as file:
		file.write("Ideal probs: \n")
		for prob in ideal_probs:
			file.write(str(prob) + " ")
		file.write('\nApprox probs:\n' + str(USC.test_srbb_method(device, args.n_qubits)(U_approx, False)))
		file.write('\nHellinger dist: ' + str(losses.hellinger(ideal_probs, USC.test_srbb_method(device, args.n_qubits)(U_approx, False))))

		if args.state:
			file.write('\nIdeal State:\n' + str(state))
			file.write('\nApprox State:\n' + str(USC.test_srbb_method(device, args.n_qubits)(U_approx_phase @ U_approx, True)))
			file.write('\nTrace:\n' + str(qml.math.trace_distance(dm, USC.test_srbb_method(device, args.n_qubits)(U_approx_phase @ U_approx, True, True))))
		if not args.only_test:
			file.write('\ntime: ' + str(elapsed_time))
		file.write('\n\n')
		

def callback(params):
	"""
		Used with Nelder Mead to print parameters

		Args:
			params (np.array): trained theta angles
	"""
	print("Current params:", params)

def cost(params, X, Y_ideal, n_qubit, loss, SU_ideal, x_max, rot_count, U_ideal, listSU, opt, state, device):
	"""
	Choose which cost function to calculate
	
	Args:
		params (np.array): List of theta parameter that are trained
		X (): the training dataset
		Y_ideal (): the ideal density matrices
		n_qubit (int): number of qubits
		loss (str): Which loss to use
		SU_ideal (np.array): the ideal matrix
		x_max (int): number of ProdT_factors and M_factors
		rot_count (int): number of rot 
		U_ideal (np.array): the approximated matrix
		listSU (): list of all 2^n_qubits possibile SU
		opt (str): selected otpimizer: 'Adam' or 'Nelder_Mead'
		state (bool): if true train the phase, if false train the modulo

	Returns:
		loss: the loss value 
	
	"""
    
	if loss == 'trace_distance' or loss == 'fidelity':
		
		predictions = [USC.make_circuit_qnode(device, n_qubit)(params, x_max, rot_count, X[i], [], state) for i in range(len(X))]
		
		predictions, states, probs = zip(*predictions)
		predictions, states, probs = list(predictions), list(states), list(probs)
		
		if loss == 'trace_distance':
			return losses.traceDistanceLoss(Y_ideal, predictions)
		elif loss == 'fidelity':
			return losses.fidelityLoss(Y_ideal, predictions)
	elif loss == 'frob':
		return losses.frobeniusLoss(params, x_max, SU_ideal, rot_count, U_ideal, listSU, opt, state, n_qubit, device)

def training(X_train, Y_ideal, batch_size, learning_rate, epochs, loss, optimizer, num_layer, n_qubit, listdet, listSU, SU_ideal, dir, x_max, U_ideal, device, circuit, pathName, path, state, args):	
	"""
	Execute the training of the neural network
	Args:
		X_train (): the training dataset
		Y_ideal (): the ideal density matrices
		batch_size (int): the batch size
		learning_rate (float): the learning rate value
		epochs (int): how many epoch of training
		loss (str): which loss function to use
		optimizer (str): which optimizer to use (Adam or Nelder_Mead)
		num_layer (int): number of layer of VQC
		n_qubit (int): number of qubits
		listdet (): list of determinant roots of the different 2^n possibilities of SU
		listSU (): list of all 2^n_qubits possibile SU
		SU_ideal (np.array): the ideal matrix
		dir (str): directory where to save/load file
		x_max (int): number of ProdT_factors and M_factors
		U_ideal (np.array): the approximated matrix
		device (str): device used
		circuit (): which circuit to approximate
		rot_count (int): number of rot 
		pathName (str):Name given by the options chosen in arguments
		path (str): where to save the files named with pathName
		state (bool): if true train the phase, if false train the modulo

	Returns
		params (): trained parameters
	    U_approx (np.array): approximated U
		cost_new (float): the value of the cost function
		norm[np.argmin(norm)] (float): the right norm value
		SU_approx (np.array): the approximated SU
		listSU[np.argmin(norm)] (np.array): the right SU
		rot_count (int): number of rotations
	"""

	params, rot_count=USC.Theta_array_gen(state, n_qubit)
	#print(params)

	if optimizer=='Adam':
		opt=qml.AdamOptimizer(stepsize=learning_rate)
	elif optimizer=='GDO':
		opt=qml.GradientDescentOptimizer(stepsize=learning_rate)
	else:
		opt=qml.QNGOptimizer(stepsize=learning_rate)

	if optimizer=='Nelder_Mead':
		params=scipy.optimize.fmin(func=losses.frobeniusLoss, x0=params, args=(x_max, SU_ideal, rot_count, U_ideal, listSU, optimizer, state, n_qubit, device), xtol=10**(-15), ftol=10**(-15),maxiter=10**20,maxfun=10**20)
		
		print(params)
		cost_new=0.0

		if not state:
			fileParams = open(path + 'paramsModulo' + pathName + '.obj', 'wb')
		else:
			fileParams = open(path + 'paramsPhase' + pathName + '.obj', 'wb')
							
		pickle.dump(params, fileParams)
		fileParams.close()

		SU_approx = qml.matrix(USC.make_circuit_qnode(device, args.n_qubits))(params, x_max, rot_count, [], [], state)
		utils.format_matrix(SU_approx)
		
	else:
		for e in range(epochs):
			print("EPOCH: ", e)
			for b in range(0, len(X_train), batch_size):
				if (b + batch_size) <= len(X_train):
					X_batch = [X_train[i] for i in range(b, b + batch_size)]
					Y_batch = [Y_ideal[i] for i in range(b, b + batch_size)]
				else:
					X_batch = [X_train[i] for i in range(b, len(X_train))]
					Y_batch = [Y_ideal[i] for i in range(b, len(X_train))]
					
				if optimizer == 'Adam':
					params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, n_qubit, loss, SU_ideal, x_max, rot_count, U_ideal, listSU, optimizer, state, device),
																params)
			#print(params)
			#input()
			print("iteration: ", e, " cost: ", cost_new)
			
			
			SU_approx = qml.matrix(USC.make_circuit_qnode(device, args.n_qubits))(params, x_max, rot_count, [], [], state)
			#format_matrix(SU_approx)
			if not state:
				fileParams = open(path + 'paramsModulo' + pathName + '.obj', 'wb')
			else:
				fileParams = open(path + 'paramsPhase' + pathName + '.obj', 'wb')
							
			pickle.dump(params, fileParams)
			fileParams.close()
			
			
			
	"""
	with open(dir + 'SU_Results' + '.txt', "a") as file:
		file.write("\nAPPROX SU:\n")
		file.write(format_matrix2(SU_approx))
	"""
	if state:
		#choose the right SU or U
		DELTA = []

		for SU in listSU:
			DELTA.append(SU - SU_approx)

		norm = []
		for d in DELTA:
			norm.append(cmath.sqrt(max(np.linalg.eig(((np.conj(d)).T) @ d)[0])))
			
		print("NORM:")
		for n in norm:
			print(n)

		print("\n DET:")
		for d in listdet:
			print(d)
		

		
		#print(listSU[np.argmin(norm)])	
		#input()
		U_approx = SU_approx * listdet[np.argmin(norm)]

			
		
		print(" ")
		#format_matrix(U_approx)

		
		fileu_app = open(path + 'U_appPhase' + pathName + '.obj', 'wb')
		file_rot_count = open(path + 'rot_countPhase' + pathName + '.obj', 'wb')
					
		pickle.dump(U_approx, fileu_app)
		fileu_app.close()

		
						
		pickle.dump(rot_count, file_rot_count)
		file_rot_count.close()
	

		return params, U_approx, cost_new, norm[np.argmin(norm)], SU_approx, listSU[np.argmin(norm)], rot_count
	else:
	
		fileu_app = open(path + 'U_appModulo' + pathName + '.obj', 'wb')
		file_rot_count = open(path + 'rot_countModulo' + pathName + '.obj', 'wb')

		pickle.dump(SU_approx, fileu_app)
		fileu_app.close()

		
						
		pickle.dump(rot_count, file_rot_count)
		file_rot_count.close()

		return params, None, cost_new, None, SU_approx, None, rot_count
