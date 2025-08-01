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

def cnoth(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()        #layer A, component 1
        A2 = qml.Hadamard.compute_matrix()    #layer A, component 2
        matrix[:] = np.kron(A1, A2)
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

def cnoti(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()
        A2 = np.eye(2, dtype=int)
        matrix[:] = np.kron(A1, A2)
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

def icnot_rev(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = np.eye(2)
        A2 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        matrix[:] = np.kron(A1, A2)
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

def cnot02(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        block1 = np.eye(4)
        block2 = np.zeros((4,4))
        block3 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        matrix[:] = np.block([[block1, block2], [block2, block3]])
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

def cnot20(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        block1 = np.array([[1, 0], [0, 0]])
        block2 = np.array([[0, 0], [0, 1]])
        block3 = np.zeros((2,2))
        matrix[:] = np.block([[block1, block3, block2, block3],
                            [block3, block1, block3, block2],
                            [block2, block3, block1, block3],
                            [block3, block2, block3, block1]])
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

def cnotx(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()
        A2 = qml.PauliX.compute_matrix()
        matrix[:] = np.kron(A1, A2)
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

def cnoty(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()
        A2 = qml.PauliY.compute_matrix()
        matrix[:] = np.kron(A1, A2)
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

def cnotz(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()
        A2 = qml.PauliZ.compute_matrix()
        matrix[:] = np.kron(A1, A2)
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

def xxx(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.PauliX.compute_matrix()
        matrix[:] = np.kron(np.kron(A1, A1), A1)
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
def xyx(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.PauliX.compute_matrix()
        A2 = qml.PauliY.compute_matrix()
        matrix[:] = np.kron(np.kron(A1, A2), A1)
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
def xyz(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.PauliX.compute_matrix()
        A2 = qml.PauliY.compute_matrix()
        A3 = qml.PauliZ.compute_matrix()
        matrix[:] = np.kron(np.kron(A1, A2), A3)
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

def hhh(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.Hadamard.compute_matrix()
        matrix[:] = np.kron(np.kron(A1, A1), A1)
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

def icnot_cnoti(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()
        A2 = np.eye(2)
        A = np.kron(A1, A2)             #layer A
        B1 = np.eye(2)
        B2 = qml.CNOT.compute_matrix()
        B = np.kron(B1, B2)             #layer B
        matrix[:] = A @ B
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

def icnotrev_cnotrevi(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        A2 = np.eye(2)
        A = np.kron(A1, A2)
        B1 = np.eye(2)
        B2 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        B = np.kron(B1, B2)
        matrix[:] = A @ B
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

def cnot02_icnot(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = np.eye(2)
        A2 = qml.CNOT.compute_matrix()
        A = np.kron(A1, A2)
        block1 = np.eye(4)
        block2 = np.zeros((4,4))
        block3 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        B = np.block([[block1, block2], [block2, block3]])
        matrix[:] = A @ B
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

def toffoli(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        matrix[:] = qml.Toffoli.compute_matrix()
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

def c0c0x(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        matrix[:] = qml.MultiControlledX.compute_matrix([0, 1], [0, 0])
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

def c1c0roty(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        U = qml.RY(np.pi/4, wires = [2])
        matrix[:] = qml.matrix(qml.ControlledQubitUnitary(U, control_wires = [0, 1], wires = [2], control_values = [1, 0]))
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

def xcnot_cnoty(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()
        A2 = qml.PauliY.compute_matrix()
        A = np.kron(A1, A2)
        B1 = qml.PauliX.compute_matrix()
        B2 = qml.CNOT.compute_matrix()
        B = np.kron(B1, B2)
        matrix[:] = A @ B
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

def hhh_xyx(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.PauliX.compute_matrix()
        A2 = qml.PauliY.compute_matrix()
        A = np.kron(np.kron(A1, A2), A1)
        B1 = qml.Hadamard.compute_matrix()
        B = np.kron(np.kron(B1, B1), B1)
        matrix[:] = A @ B
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

def hhh_xyz(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.PauliX.compute_matrix()
        A2 = qml.PauliY.compute_matrix()
        A3 = qml.PauliZ.compute_matrix()
        A = np.kron(np.kron(A1, A2), A3)
        B1 = qml.Hadamard.compute_matrix()
        B = np.kron(np.kron(B1, B1), B1)
        matrix[:] = A @ B
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

def hii_cnoti_ihi_icnot(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.Hadamard.compute_matrix()
        A2 = np.eye(4)
        A = np.kron(A1, A2)
        B1 = qml.CNOT.compute_matrix()
        B2 = np.eye(2)
        B = np.kron(B1, B2)
        C1 = np.eye(2)
        C2 = qml.Hadamard.compute_matrix()
        C3 = np.eye(2)
        C = np.kron(np.kron(C1, C2), C3)
        D1 = np.eye(2)
        D2 = qml.CNOT.compute_matrix()
        D = np.kron(D1, D2)
        matrix[:] = A @ B @ C @ D
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

def hhh_xxx(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.Hadamard.compute_matrix()
        A = np.kron(np.kron(A1, A1), A1)
        B1 = qml.PauliX.compute_matrix()
        B = np.kron(np.kron(B1, B1), B1)
        matrix[:] = A @ B
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

def hxx_iyz(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = np.eye(2)
        A2 = qml.PauliY.compute_matrix()
        A3 = qml.PauliZ.compute_matrix()
        A = np.kron(np.kron(A1, A2), A3)
        B1 = qml.Hadamard.compute_matrix()
        B2 = qml.PauliX.compute_matrix() 
        B = np.kron(np.kron(B1, B2), B2)
        matrix[:] = A @ B
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
            


def swapi_iswap(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = np.eye(2)
        A2 = qml.SWAP.compute_matrix()
        A = np.kron(A1, A2)
        B1 = qml.SWAP.compute_matrix()
        B2 = np.eye(2)
        B = np.kron(B1, B2)
        matrix[:] = A @ B
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

def iswapi_iswap(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = np.eye(2)
        A2 = qml.SWAP.compute_matrix()
        A = np.kron(A1, A2)
        B1 = qml.ISWAP.compute_matrix()
        B2 = np.eye(2)
        B = np.kron(B1, B2)
        matrix[:] = A @ B
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

def rootnotii_ihh_ycs(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        A1 = qml.PauliY.compute_matrix()
        A2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, +1j]])
        A = np.kron(A1, A2)
        B1 = np.eye(2)
        B2 = qml.Hadamard.compute_matrix()
        B = np.kron(np.kron(B1, B2), B2)
        C1 = 1/2 * np.array([[1 + 1j, 1 -1j], [1 - 1j, 1 + 1j]])
        C2 = np.eye(4)
        C = np.kron(C1, C2)
        matrix[:] = A @ B @ C
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

def qft3(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        matrix[:] = qml.QFT.compute_matrix(3)
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

def grover3(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        matrix[:] = qml.GroverOperator.compute_matrix(3, 0)
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

def fredkin(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        matrix[:] = qml.CSWAP.compute_matrix()
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
    
def ccx(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        matrix[:] = qml.MultiControlledX.compute_matrix(control_wires  = [0, 1], control_values  = [1, 1])
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

def orCircuit(dev, n_qubit):
        @qml.qnode(dev)
        def circuit(x, matrix, device_type, state=False):
            if len(x) != 0:
                AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
            A = np.kron(np.kron(qml.PauliX.compute_matrix(), qml.PauliX.compute_matrix()), qml.PauliX.compute_matrix())
            B = np.kron(np.kron(qml.PauliX.compute_matrix(), qml.PauliX.compute_matrix()), np.eye(2))
            matrix[:] = B @ qml.MultiControlledX.compute_matrix([0, 1], [1, 1]) @ A
                        
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

def peres(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        matrix[:] = np.kron(np.eye(2), qml.CNOT.compute_matrix()) @ qml.MultiControlledX.compute_matrix([0, 1], [1, 1])
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

  
def deutsch_jozsa_balanced(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):  
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)

        l1 = np.kron(np.eye(4, dtype = int), qml.X.compute_matrix())
        l2 = np.kron(np.kron(qml.Hadamard.compute_matrix(), qml.Hadamard.compute_matrix()), qml.Hadamard.compute_matrix())
        #cnot 02
        block1 = np.eye(4)
        block2 = np.zeros((4,4))
        block3 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        oracle1 = np.block([[block1, block2], [block2, block3]])
        #cnot 12
        oracle2 = np.kron(np.eye(2, dtype = int), qml.CNOT.compute_matrix())
        l3 = np.kron(np.kron(qml.Hadamard.compute_matrix(), qml.Hadamard.compute_matrix()), np.eye(2, dtype = int))

        matrix[:] = l3 @ oracle2 @ oracle1 @ l2 @ l1
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


def ghz(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type, state=False):
        if len(x) != 0:
            AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        
        l1 = np.kron(qml.Hadamard.compute_matrix(), np.eye(4, dtype = int))
        l2 = np.kron(qml.CNOT.compute_matrix(), np.eye(2, dtype = int))
        l3 = np.kron(np.eye(2, dtype = int), qml.CNOT.compute_matrix())

        matrix[:] = l3 @ l2 @ l1
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