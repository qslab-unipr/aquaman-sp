import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
import numpy as np
import cmath
import config


def random(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, phase=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        if phase:
            qml.QubitUnitary(SU, wires = range(n_qubit))
        else:
            qml.QubitUnitary(matrix, wires = range(n_qubit))

        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def entangled4(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = np.eye(4)
        A2 = qml.CNOT.compute_matrix()
        A = np.kron(A1, A2)
        B1 = np.eye(4)
        B2 = qml.Hadamard.compute_matrix()
        B3 = np.eye(2)
        B = np.kron(np.kron(B1, B2), B3)
        C1 = np.eye(2)
        C2 = qml.CNOT.compute_matrix()
        C = np.kron(np.kron(C1, C2), C1)
        D1 = np.eye(2)
        D2 = qml.Hadamard.compute_matrix()
        D3 = np.eye(4)
        D = np.kron(np.kron(D1, D2), D3)
        E1 = qml.CNOT.compute_matrix()
        E2 = np.eye(4)
        E = np.kron(E1, E2)
        F1 = qml.Hadamard.compute_matrix()
        F2 = np.eye(8)
        F = np.kron(F1, F2)
        matrix[:] = A @ B @ C @ D @ E @ F
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        if state:
            qml.QubitUnitary(matrix, wires = range(n_qubit))
            return qml.state()
        else:
            qml.QubitUnitary(SU, wires = range(n_qubit))
            if device_type == "sim":
                return qml.density_matrix(wires = range(n_qubit))
            else:
                return qml.probs(wires=range(n_qubit))
    return circuit

def deutsch_jozsa_unbalanced(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)

        l1 = np.kron(np.eye(8, dtype = int), qml.X.compute_matrix())
        l2 = np.kron(np.kron(np.kron(qml.Hadamard.compute_matrix(), qml.Hadamard.compute_matrix()), qml.Hadamard.compute_matrix()), qml.Hadamard.compute_matrix())
        oracle = np.kron(np.eye(8, dtype = int), qml.X.compute_matrix())
        l3 = np.kron(np.kron(np.kron(qml.Hadamard.compute_matrix(), qml.Hadamard.compute_matrix()), qml.Hadamard.compute_matrix()), np.eye(2, dtype = int))

        matrix[:] = l3 @ oracle @ l2 @ l1
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        if state:
            qml.QubitUnitary(matrix, wires = range(n_qubit))
            return qml.state()
        else:
            qml.QubitUnitary(SU, wires = range(n_qubit))
            if device_type == "sim":
                return qml.density_matrix(wires = range(n_qubit))
            else:
                return qml.probs(wires=range(n_qubit))
        return circuit

def deutsch_jozsa_balanced(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False): 
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)

        l1 = np.kron(np.eye(8, dtype = int), qml.X.compute_matrix())
        l2 = np.kron(np.kron(np.kron(qml.Hadamard.compute_matrix(), qml.Hadamard.compute_matrix()), qml.Hadamard.compute_matrix()), qml.Hadamard.compute_matrix())
        oracle = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],])
        l3 = np.kron(np.kron(np.kron(qml.Hadamard.compute_matrix(), qml.Hadamard.compute_matrix()), qml.Hadamard.compute_matrix()), np.eye(2, dtype = int))

        matrix[:] = l3 @ oracle @ l2 @ l1
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        if state:
            qml.QubitUnitary(matrix, wires = range(n_qubit))
            return qml.state()
        else:
            qml.QubitUnitary(SU, wires = range(n_qubit))
            if device_type == "sim":
                return qml.density_matrix(wires = range(n_qubit))
            else:
                return qml.probs(wires=range(n_qubit))
    return circuit

def hhhh(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):    
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.Hadamard.compute_matrix()
        matrix[:] = np.kron(np.kron(np.kron(A1, A1), A1), A1)
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
        if state:
            qml.QubitUnitary(matrix, wires = range(n_qubit))
            return qml.state()
        else:
            qml.QubitUnitary(SU, wires = range(n_qubit))
            if device_type == "sim":
                return qml.density_matrix(wires = range(n_qubit))
            else:
                return qml.probs(wires=range(n_qubit))
    return circuit

