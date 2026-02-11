import pennylane as qml
import numpy as np
import autograd.numpy as anp
import qsp_srbb_circuit as USC
import utils
import cmath

def traceDistanceLoss(Y_ideal, predictions):
	"""
	Calculate the trace distance between the ideal states and the states produced by the QNN
	
	Args:
		Y_ideal (): List of ideal density matrices
		predictions (): List of density matrices produced by the network

	Returns:
		loss (float): the mean of the trace
	"""
	num_samples = len(predictions)
	loss = 0.0
	for i in range(len(predictions)):
		loss += qml.math.trace_distance(predictions[i], Y_ideal[i])

	return loss / num_samples

def fidelityLoss(Y_ideal, predictions):
	"""
	Calculate the fidelity between the ideal states and the states produced by the QNN
	
	Args:
		Y_ideal (): List of ideal density matrices
		predictions (): List of density matrices produced by the network

	Returns
		loss (float): the mean of the fidelities
	"""
	num_samples = len(predictions)
	loss = 0.0
	for i in range(len(predictions)):
		loss += 1 - qml.math.fidelity(predictions[i], Y_ideal[i])

	return loss / num_samples

def hellingerLoss(prob1, prob2):
	"""
	Calculate the Hellinger distance between 2 probabilities distribution

	Args:	
		prob1 (np.array): list of ideal probabilities
		prob2 (np.array): list of probabilities produced by the QNN

	Returns:
		distance (float): hellinger distance
	"""
	num_samples = len(prob2)
	loss = 0.0
	for i in range(num_samples):
		loss += (1/anp.sqrt(2)) * anp.sqrt(anp.sum((anp.sqrt(prob1[i]) - anp.sqrt(prob2[i]))**2))
	return loss /num_samples

def hellinger(prob1, prob2):
	"""
	Calculate the Hellinger distance between 2 probabilities distribution

	Args:	
		prob1 (np.array): list of ideal probabilities
		prob2 (np.array): list of probabilities produced by the QNN

	Returns:
		distance (float): hellinger distance
	"""
	distance = (1/np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(prob1) - np.sqrt(prob2))**2))
	return distance

def frobeniusLoss(params, x_max, SU_ideal, rot_count, U_ideal, listSU, opt, state, n_qubit, dev, initial_state):
	"""
	Calculate the frobenius Norm between the ideal matrix and the one produced by the VQC

	Args:
		params (np.array): List of theta parameter that are trained
		x_max (int): number of ProdT_factors and M_factors
		SU_ideal (np.array): the ideal matrix
		rot_count (int): number of rot 
		U_ideal (np.array): the approximated matrix
		listSU (): list of all 2^n_qubits possibile SU
		opt (str): Selected optimizer: 'Adam' or 'Nelder_Mead'
		state (bool): if true train the phase, if false train the modulo 

	Returns
	  	loss (float): the frobenius norm value 
	"""
	
	if not state:
		Msub = U_ideal - qml.matrix(USC.make_circuit_qnode(dev, n_qubit), wire_order=range(0, n_qubit))(params, x_max, rot_count, [], [], initial_state)
	else:
		SU_ideal = calculate_closer_SU(listSU, qml.matrix(USC.make_circuit_qnode(dev, n_qubit), wire_order=range(0, n_qubit))(params, x_max, rot_count, [], [], initial_state, True), SU_ideal, opt)
		Msub = SU_ideal - qml.matrix(USC.make_circuit_qnode(dev, n_qubit), wire_order=range(0, n_qubit))(params, x_max, rot_count, [], [], initial_state, True)
	Msub_H = (np.conj(Msub)).T
	
	return np.real(np.trace(np.dot(Msub, Msub_H)) ** 0.5)

def calculate_closer_SU(listSU, SU_approx, previous_SU, opt):
	"""
	choose the right SU or U for frobenius loss
	
	Args:
		listSU (): list of possible SU
		SU_approx (np.array): the approximated SU
		previous_SU (np.array): ideal SU of previous step
		opt (str): selected optimizer
	
	Returns:
		ideal_SU (np.array): the ideal SU
	"""
	DELTA = []

	for SU in listSU:
		DELTA.append(SU - SU_approx)


	norm = []
	for d in DELTA:
		if opt == 'Adam':
			norm.append(cmath.sqrt(max(np.linalg.eig(((np.conj(d._value)).T) @ d._value)[0])))
		else:
			norm.append(cmath.sqrt(max(np.linalg.eig(((np.conj(d)).T) @ d)[0])))
			
	if np.min(norm) < 10:
		return listSU[np.argmin(norm)]
	else:
		return previous_SU