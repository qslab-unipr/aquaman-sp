import numpy as np

def create_dataset(args):
	"""
	Create the training dataset if fidelity or trace distance are selected as loss function
	
	Args:
		args (): given arguments

	Returns:
		X_train (): training set
		X_test (): test set
	"""
	#Step 1: create a normalized complex random vector for training
	X_train=[]
	for i in range(args.num_samples):
		real_part=np.random.normal(size=2**args.n_qubits)
		imag_part=np.random.normal(size=2**args.n_qubits)
		random_complex_vector=real_part+1j*imag_part
		norm=np.linalg.norm(random_complex_vector)
		normalized_vector=random_complex_vector/norm
		X_train.append(normalized_vector)

	#Step2: create a normalized complex random vector for testing
	X_test=[]
	for i in range(args.test_size):
		real_part=np.random.normal(size=2**args.n_qubits)
		imag_part=np.random.normal(size=2**args.n_qubits)
		random_complex_vector=real_part+1j*imag_part
		norm=np.linalg.norm(random_complex_vector)
		normalized_vector=random_complex_vector/norm
		X_test.append(normalized_vector)
	return X_train,X_test