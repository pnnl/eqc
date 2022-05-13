    import numpy as np
import qiskit
import sys
import random
import matplotlib.pyplot as plt
import os
import json
from TrainableCircuit import TrainableCircuit
from Optimizer import Adam,SGD

class VQE(TrainableCircuit):
    def __init__(self,k,entanglement='full',seed=43):
        super().__init__(k,entanglement,seed)
        self.gradient_circuits = None
        np.random.seed(seed)
        self.seed = seed
        self.backend = None
        self.parameters = None
        self.CIRCUITS_TO_INDUCE = None
        self.qubits = qiskit.circuit.QuantumRegister(k)
        self.circuit = qiskit.circuit.QuantumCircuit(self.qubits)
        self.entanglement_mode = entanglement
        self.hamiltonian = np.zeros((2**self.k,2**self.k),dtype="complex128")
        self.four_qubit_heisenberg()

    def four_qubit_heisenberg(self):
        connections = [(0, 1), (0, 3), (1, 2), (2, 3)]
        Z = self.PAULI_MATRICES["Z"]
        X = self.PAULI_MATRICES["X"]
        Y = self.PAULI_MATRICES["Y"]
        I = self.PAULI_MATRICES["I"]
        for pair in connections:
            z = 1
            x = 1
            y = 1
            for i in range(len(self.qubits)):
                if i in pair:
                    z = np.kron(z, Z)
                    x = np.kron(x, X)
                    y = np.kron(y, Y)
                else:
                    z = np.kron(z, I)
                    x = np.kron(x, I)
                    y = np.kron(y, I)
            z = z.astype('complex128')
            x = x.astype('complex128')
            y = y.astype('complex128')
            self.hamiltonian += z + x + y
        for qub in range(len(self.qubits)):
            h = 1
            for i in range(len(self.qubits)):
                if i == qub:
                    h = np.kron(h, Z)
                else:
                    h = np.kron(h, I)
            h = h.astype('complex128')
            self.hamiltonian += h
        print(self.hamiltonian)



    def initialize_random_hamiltonian_matrix(self,pauli_model=True):
        if pauli_model is True:
            weights = np.random.random(10)
            for weight in weights:
                new_matrix = 1
                for i in range(self.k):
                    new_matrix = np.kron(new_matrix, self.PAULI_MATRICES[self.z_or_i()])
                self.hamiltonian += new_matrix*weight*(1/len(weights))
        else:
            self.hamiltonian = np.random.rand(2**self.k,2**self.k)
            self.hamiltonian *= 10

    def z_or_i(self):
        p=0.5
        if random.random() > p:
            return "Z"
        else:
            return "I"

    def initialize(self):
        if self.parameters is None:
            self.parameters = []
            for qub in self.qubits:
                self.circuit.ry(self.new_param(),qub)
                self.circuit.rz(self.new_param(),qub)
            for q1,q2 in zip(list(self.qubits)[:-1],list(self.qubits)[1:]):
                self.circuit.cx(q1,q2)
            self.circuit.cx(self.qubits[-1],self.qubits[0])
            for qub in self.qubits:
                self.circuit.ry(self.new_param(),qub)
                self.circuit.rz(self.new_param(),qub)
            self.circuit.measure_all()
            self.initialize_params()
        else:
            print("Model already initialized - Parameters exist.")

    def set_backend(self,backend):
        self.backend = backend

    def cost_function(self,statevector):
        return np.matmul(np.matmul(statevector.T,self.hamiltonian),statevector)

