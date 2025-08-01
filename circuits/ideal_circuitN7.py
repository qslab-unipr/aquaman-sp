import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
import numpy as np
import cmath
import config

n_qubit = 7

dev = qml.device('default.qubit', wires = n_qubit)
#dev = config.dev

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

