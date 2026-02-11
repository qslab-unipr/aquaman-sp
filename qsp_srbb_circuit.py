import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
#import matplotlib.pyplot as plt
from pennylane.templates.embeddings import AmplitudeEmbedding
#from pennylane_qiskit.remote import RemoteDevice

def make_circuit_qnode(dev, n_qubit, device_type='sim'):
	@qml.qnode(dev)
	def circuit(params, x_max, rot_count, x=[], U_approx=[], initial_state = False, state = False):
		"""
		Create the circuit from gray code

		Args:
			params (np.array): theta angles
			x_max (int): number of ProdT_factors and M_factors
			rot_count (int): num of rots
			x (): state to be encoded. if trace or fidelity are used as loss function
			U_approx (np.array): approximated U
			state (bool): if false train the modulo, if true train the phase 
		"""

		simpli_Gray_collection=Simplification_Gray_matrix(Gray_matrix(n_qubit-1,False), n_qubit)

		#fig, ax = qml.draw_mpl(SU_approx_circuit)(params, x_max, simpli_Gray_collection, state, n_qubit, initial_state)
		#fig.show()
		#input()
		#if there is no U, then i want to train the network to achieve it
		if len(U_approx) == 0:
			#if fidelity or trace distance are used as loss function, encodes the state
			if len(x) != 0:
				AmplitudeEmbedding(x, range(n_qubit), normalize = True)
				#qml.Barrier(wires = [0, 1])
			
			#add the VQC
			SU_approx_circuit(params, x_max, simpli_Gray_collection, state, n_qubit, initial_state)
			if device_type == 'hw':
				return [
					qml.expval(qml.PauliZ(0)),           # <Z ⊗ I>
					qml.expval(qml.PauliZ(1)),           # <I ⊗ Z>
					qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))  # <Z ⊗ Z>
				]
			else:
				if len(x) == 0:
					return qml.density_matrix(range(n_qubit))
				else:
					return qml.density_matrix(range(n_qubit)), qml.state(), qml.probs(range(n_qubit))
			
		else: #testing
			if len(x) != 0:
				AmplitudeEmbedding(x, range(n_qubit), normalize = True)

			#based on if the function has the parameters or the matrix, generate the circuit
			if len(params) == 0:
				qml.QubitUnitary(U_approx, wires = range(n_qubit))
			else:
				SU_approx_circuit(params, x_max, simpli_Gray_collection, state, n_qubit, initial_state)			

			if isinstance(dev, RemoteDevice):
				return qml.counts(wires = range(n_qubit)), qml.counts(wires =range(n_qubit)), qml.probs(range(n_qubit))
			else:
				return qml.state(), qml.probs(range(n_qubit)), qml.density_matrix(range(n_qubit))
	return circuit
	

def amplitude_density(dev, n_qubit):
	@qml.qnode(dev)
	def circuit(x):
		"""
		Retrieve Density matrix of state

		Args:
			x (): state to be encoded

		Returns:
			(): density matrix
		"""
		AmplitudeEmbedding(x, range(n_qubit), normalize = True)
		return qml.density_matrix(range(n_qubit))
	return circuit


#Format useful for checks
def format_matrix(matrix):
	M=('\n'.join(''.join('{:11}'.format(np.round(item,1)) for item in row) for row in matrix))
	return M


def Gray_matrix(n_bits,side=bool):
	"""
		Used for ZETA_factor
		
		Args:
			n_bits (int): number of bits
			side (bool): True for right decomposition, False for left decomposition

		Returns:
			(): gray matrix collection
	"""
	Gray_matrix_collection=[]
	if side==True:
		Gray_matrix_1=np.array([[0],[1]])
		Gray_matrix_collection.append(Gray_matrix_1)
		Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
		for i in range(2,n_bits+1):
			Gray_matrix_next=np.empty(shape=(2**i,i),dtype=int)
			semi_rows=2**(i-1)
			Gray_matrix_next[0:semi_rows,0]=[0]*semi_rows
			Gray_matrix_next[0:semi_rows,1:i]=Gray_matrix_1
			Gray_matrix_next[semi_rows:2**i,0]=[1]*semi_rows
			Gray_matrix_next[semi_rows:2**i,1:i]=Gray_matrix_1reverse
			Gray_matrix_collection.append(Gray_matrix_next)
			Gray_matrix_1=Gray_matrix_next
			Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
	else:
		Gray_matrix_1=np.array([[1],[0]])
		Gray_matrix_collection.append(Gray_matrix_1)
		Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
		for i in range(2,n_bits+1):
			Gray_matrix_next=np.empty(shape=(2**i,i),dtype=int)
			semi_rows=2**(i-1)
			Gray_matrix_next[0:semi_rows,0]=[1]*semi_rows
			Gray_matrix_next[0:semi_rows,1:i]=Gray_matrix_1
			Gray_matrix_next[semi_rows:2**i,0]=[0]*semi_rows
			Gray_matrix_next[semi_rows:2**i,1:i]=Gray_matrix_1reverse
			Gray_matrix_collection.append(Gray_matrix_next)
			Gray_matrix_1=Gray_matrix_next
			Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
	return Gray_matrix_collection


def Simplification_Gray_matrix(Gray_matrix_collection, n_qubit):
	"""
		Inside ZETA_factor, Simplify the CNOT number
		CNOT parameters depend on this matrix

		Args:
			Gray_matrix_collection ():

		Returns:
			(): simplified gray
	"""
	simpli_Gray_collection=[]
	for p in range(n_qubit-1):
		Gray_matrix_p=Gray_matrix_collection[p]
		extension=Gray_matrix_p[0]
		Gray_matrix_extended=np.empty(shape=(2**(p+1)+1,p+1))
		Gray_matrix_extended[0:2**(p+1),0:p+1]=Gray_matrix_p
		Gray_matrix_extended[2**(p+1),0:p+1]=extension
		simpli_Gray_matrix=np.empty(shape=(2**(p+1),p+1),dtype=int)
		for i in range(2**(p+1)):
			m1=Gray_matrix_extended[i]
			m2=Gray_matrix_extended[i+1]
			for j in range(p+1):
				if m1[j]==m2[j]:
					simpli_Gray_matrix[i][j]=0
				else:
					simpli_Gray_matrix[i][j]=1
		simpli_Gray_collection.append(simpli_Gray_matrix)
	
	return simpli_Gray_collection


def Theta_array_gen(state, n_qubit):
	"""
	Generates the total number of parameters for the circuit

	Args:
		state (bool): True if train phase, False if train modulo

	Returns:
		params (np.array): trainable parameters
		rot_count_ZETA (int): number of rot 
	"""

	#ZETA FACTOR -> exp{3,8,15,24...}
	if not state:
		rot_count_ZETA = 1
		for n in range(2, n_qubit+1):
			rot_count_ZETA += 2**n - 1
	else:
		rot_count_ZETA = 2**n_qubit -1

	params=pnp.random.randn(rot_count_ZETA, requires_grad=True)
	return params,rot_count_ZETA

def ZETA_factor2qubits(theta_collection_ZETA,delete=bool, modulo = True):
	"""
	ZETA_factor definition for 2 qubits (special case outside the scalability scheme)

	Args:
		theta_collection_ZETA (): theta parameters
		delete (bool)
		modulo (bool): if modulo add two final gates
	"""
	if modulo:
		qml.QubitUnitary([[1, 0], [0, -1j]], wires = 1)
		qml.Hadamard(1)
	if delete==False:
		qml.CNOT(wires=[0,1])
		qml.RZ(theta_collection_ZETA[0],wires=[1])
		qml.CNOT(wires=[0,1])
		qml.RZ(theta_collection_ZETA[1],wires=[0])
		qml.RZ(theta_collection_ZETA[2],wires=[1])
	else:
		qml.RZ(theta_collection_ZETA[0],wires=[1])
		qml.CNOT(wires=[0,1])
		qml.RZ(theta_collection_ZETA[1],wires=[0])
		qml.RZ(theta_collection_ZETA[2],wires=[1])
	if modulo:
		qml.Hadamard(1)
		qml.S(1)
	


def ZETA_factor(theta_collection_ZETA,simpli_Gray_collection, n, modulo = True):
	"""
	ZETA_factor definition for n>2 qubits
	
	Args:
		theta_collection_ZETA (): theta parameters
		simpli_gray_collection (): simplified gray
		n (int): qubits
		modulo (bool): true for modulo, false for phase
	"""
	offset=0
	if modulo:
		qml.QubitUnitary([[1, 0], [0, -1j]], wires = n-1)
		qml.Hadamard(n-1)
	for k in range(n - 1):
		for i in range(simpli_Gray_collection[n-2-k].shape[0]):
			for j in range(simpli_Gray_collection[n-2-k].shape[1]):
				if simpli_Gray_collection[n-2-k][i][j]==1:
					qml.CNOT(wires=[j,n-1-k])
					qml.RZ(theta_collection_ZETA[offset + i],wires=[n-1-k])		
		offset += simpli_Gray_collection[n-2-k].shape[0]		
	qml.RZ(theta_collection_ZETA[-1],wires=[0])
	if modulo:
		qml.Hadamard(n-1)
		qml.S(n-1)
	

def SU_approx_circuit(params, x_max, simpli_Gray_collection, state, n_qubit, initial_state):
	"""
		Generate the VQC

		Args:
			params (np.array): theta angles
			x_max (int): number of ProdT_factors and M_factors
			simpli_gray_collection (): simplified gray
			state (bool): true for phase circuit, false for modulo

	"""
	
	if not state:
		theta_collection_ZETA=params[1:]
		if not initial_state:
			qml.RY(params[0], wires = 0)
	else:
		theta_collection_ZETA = params


	if n_qubit==2:
		
		#VQC modulo
		if not state:
			ZETA_factor2qubits(theta_collection_ZETA[0:3],False)
		else:
			#VQC phase
			ZETA_factor2qubits(theta_collection_ZETA,False, False)
	else:
		#ZETA FACTOR
		if not state:
			previous_index = 0
			if not initial_state:
				for n in range(2, n_qubit+1):
					end_index = previous_index + 2**n - 1
					#print(n)
					theta_zeta_n = theta_collection_ZETA[previous_index: end_index]
					#print(theta_zeta_n)
					#input()
					if n == 2:
						ZETA_factor2qubits(theta_zeta_n, False)
					else:
						ZETA_factor(theta_zeta_n,simpli_Gray_collection[:n-1], n)
					previous_index = end_index
			else: #circuit inverted if |psi> -> |0>
				for n in range(n_qubit, 1, -1):
					end_index = previous_index + 2**n - 1
					#print(n)
					theta_zeta_n = theta_collection_ZETA[previous_index: end_index]
					#print(theta_zeta_n)
					#input()
					if n == 2:
						ZETA_factor2qubits(theta_zeta_n, False)
					else:
						ZETA_factor(theta_zeta_n,simpli_Gray_collection[:n-1], n)
					previous_index = end_index
		else:
			#VQC phase
			ZETA_factor(theta_collection_ZETA,simpli_Gray_collection, n_qubit, False)

	if not state and initial_state:
		qml.RY(params[0], wires = 0)

def test(dev, n_qubit):
	@qml.qnode(dev)

	def circuit(U, paramsMod, paramsPhase, state, dm, rot_countMod, rot_countPhase, x_max, hw):
		"""
		Test the optimized VQC

		Args:
			U (np.array): approximated U
			paramsMod (np.array): optimized modulo theta
			paramsPhase (np.array): optimized phase theta
			state (bool): true for precise state with phase, false for probabilities
			dm (): density matrix of state
			rot_countMod (int): number of rot for modulo
			rot_countPhase (int): number of rot for phase
			x_max (int): number of ProdT_factors and M_factors
			hw (str): true for hardware execution

		"""

		if len(U) != 0:
			qml.QubitUnitary(U, wires = range(n_qubit))
		else:
			simpli_Gray_collection=Simplification_Gray_matrix(Gray_matrix(n_qubit-1,False))
			SU_approx_circuit(paramsMod, x_max, simpli_Gray_collection, False)
			if state:
				SU_approx_circuit(paramsPhase, x_max, simpli_Gray_collection, True)			

		if state:
			if hw:
				return qml.probs(wires = range(n_qubit))
			if not dm:
				return qml.state()
			else: 
				return qml.density_matrix(wires = range(n_qubit))
		else:
			return qml.probs(wires = range(n_qubit))
	return circuit

def test_srbb_method(dev, n_qubit):
	@qml.qnode(dev)
	def circuit(U, state, U_approxIS = None, initial_state = None, dm = False):
		"""
		Test VQC

		Args:
			U (np.array): approximated U
			state (bool): true for precise state, false for probabilities
			dm (bool): True to retrieve density matrix for trace distance computation
		"""

		if initial_state is not None:
			AmplitudeEmbedding(initial_state, range(n_qubit), normalize=True)
		
		if U_approxIS is not None:
			qml.QubitUnitary(U_approxIS, wires = range(n_qubit))


		qml.QubitUnitary(U, wires = range(n_qubit))

		if state:
			if not dm:
				return qml.state()
			else: 
				return qml.density_matrix(wires = range(n_qubit))
		else:
			return qml.probs(wires = range(n_qubit))
	return circuit



def test_srbb_method_params(dev, n_qubit):
	@qml.qnode(dev)
	def circuit(paramsMod, paramsPhase, state, paramsModIS = None, paramsPhaseIS = None, initial_state = None, dm = False):
		"""
		Test VQC

		Args:
			U (np.array): approximated U
			state (bool): true for precise state, false for probabilities
			dm (bool): True to retrieve density matrix for trace distance computation
		"""

		x_max=2**(n_qubit-1)-1

		if initial_state is not None:
			AmplitudeEmbedding(initial_state, range(n_qubit), normalize=True)
		
		if paramsModIS is not None:
			simpli_Gray_collection=Simplification_Gray_matrix(Gray_matrix(n_qubit-1,False), n_qubit)
			SU_approx_circuit(paramsPhaseIS, x_max, simpli_Gray_collection, True, n_qubit, True)
			SU_approx_circuit(paramsModIS, x_max, simpli_Gray_collection, False, n_qubit, True)			
			
		
		if paramsMod is not None:
			simpli_Gray_collection=Simplification_Gray_matrix(Gray_matrix(n_qubit-1,False), n_qubit)
			SU_approx_circuit(paramsMod, x_max, simpli_Gray_collection, False, n_qubit, False)

			SU_approx_circuit(paramsPhase, x_max, simpli_Gray_collection, True, n_qubit, False)


		if state:
			if not dm:
				return qml.state()
			else: 
				return qml.density_matrix(wires = range(n_qubit))
		else:
			return qml.probs(wires = range(n_qubit))
	return circuit