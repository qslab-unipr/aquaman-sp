# SRBB-Based Quantum State Preparation

We acknowledge the financial support from Spoke 10 - ICSC - "National Research Centre in High Performance Computing, Big Data and Quantum Computing" and from PNRR MUR project PE0000023-NQSTI, both funded by European Union – NextGenerationEU.


## Description

This repository contains the source code for the paper ["SRBB-based Quantum State Preparation"](https://arxiv.org/abs/2503.13647) submitted to [CF' 25](https://www.computingfrontiers.org/2025/).

All modules have been developed in Python with the support of the PennyLane library, which enables quantum computing simulations. The repository is organised into various files:
- **main.py** is the execution file. It permits to set different parameters, such as the optimizer, the loss function, the learning rate etc..;
- **eval.py** contains the main program, which create the U and SU matrices that should be approximated by the VQC and the training and testing section;
- **losses.py** contains the different losses that can be used;
- **qsp_srbb_circuit.py** contains the variational quantum circuit;
- **config.py** sets the device where we want to run the quantum circuit;
- **dataset.py** generates the training dataset;
- **tree.py** contains a class that generate the amplitudes tree;
- **utils.py** contains helper function;
- **unitary_state_gen.py** contains functions to generate state and unitaries needed to obtain that state
- **ideal_modulesNx.py** contains various circuits that can be tested.

## How To Run
During the execution, various parameters can be set: <br />
`epochs 40` <br />
`lr 0.01` <br />
`opt 'Adam' or 'Nelder_Mead'` <br />
`loss 'trace_distance', 'fidelity' or 'frob'` <br />
`num_samples 1000`: size of the training set <br />
`num_layer 1`: number of layers of the VQC <br />
`device 'sim' or 'hw'` <br />
`n_qubits [2,...,8]` <br />
`path 'path_to_save_file'` <br />
`state Boolean`: True if we want to obtain the precise state (modulo and phase), false if we want only the probabilities (modulo)  <br />
`--initial_state 'zero' or 'random'`: 'zero' if we want to start from |0> and arrive to $|\psi\rangle$ or from another random state <br />

In the main file, you can set if you want to reach a random state or the |0> state 

The Nelder Mead optimizer can be used only with the frobenius loss.
With the Adam otpimizer 40/50 epochs are needed.

## License
This project is licensed under the MIT License – see the LICENSE file for details.
