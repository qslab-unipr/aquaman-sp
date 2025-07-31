import pennylane as qml
from qiskit import *
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:
    pass
try:
    from iqm.qiskit_iqm import IQMProvider
except ImportError:
    pass

"""
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token=''
)
"""

def set_device(n_qubit: int, device_type: str ='sim'):
    if device_type == 'sim':
        dev = qml.device('default.qubit', wires = n_qubit)
    
    elif device_type == 'hw':
        QiskitRuntimeService.save_account(channel='ibm_quantum', token='', overwrite=True, set_as_default=True)

        service = QiskitRuntimeService(channel="ibm_quantum", instance='ibm-q/open/main')
        bck = service.least_busy(operational = True, simulator = False, min_num_qubits = n_qubit)
        print(bck)
        print(bck.provider)
        dev = qml.device('qiskit.remote', wires = n_qubit, backend = bck, shots = 1024)
    elif device_type == "iqm":
        sim=qml.device('default.qubit', wires = n_qubit)
        server_url = "https://cocos.resonance.meetiqm.com/deneb"  # For example https://cocos.resonance.meetiqm.com/garnet
        api_token = "token"


        # Initialize backend
        dev = IQMProvider(server_url, token=api_token).get_backend()
    else:
        raise ValueError(f"Device type not recognized: {device_type}")
    
    return dev


"""

if device == "sim":
    dev = qml.device('default.qubit', wires = n_qubit)
elif device == "hw":
    # Or save your credentials on disk.
    #QiskitRuntimeService.save_account(channel='ibm_quantum', token='', overwrite=True, set_as_default=True)

    service = QiskitRuntimeService(channel="ibm_quantum", instance='ibm-q/open/main')

    bck = service.least_busy(operational = True, simulator = False, min_num_qubits = n_qubit)
    print(bck)
    #print(bck.provider)
    dev = qml.device('qiskit.remote', wires = n_qubit, backend = bck, shots = 1024)
elif device == "iqm":
    sim=qml.device('default.qubit', wires = n_qubit)
    server_url = "https://cocos.resonance.meetiqm.com/deneb"  # For example https://cocos.resonance.meetiqm.com/garnet
    api_token = "token"


    # Initialize backend
    dev = IQMProvider(server_url, token=api_token).get_backend()
"""