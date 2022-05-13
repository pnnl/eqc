import numpy as np
import qiskit
import sys
import random
import matplotlib.pyplot as plt
import os
import json
from TrainableCircuit import TrainableCircuit
from Optimizer import Adam,SGD

class QAOA(TrainableCircuit):
    def __init__(self,k,entanglement='full',seed=45):
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
        self.hamiltonian = np.zeros((2**self.k,2**self.k))
        self.initialize_hamiltonian()


    def initialize_hamiltonian(self):
        self.connections = [(0,1),(0,3),(1,2),(2,3)]
        Z = self.PAULI_MATRICES["Z"]
        I = self.PAULI_MATRICES["I"]
        for pair in self.connections:
            h = 1
            for i in range(len(self.qubits)):
                if i in pair:
                    h = np.kron(h,Z)
                else:
                    h = np.kron(h,I)
            self.hamiltonian += 0.5*(np.identity(h.shape[0])-h)
        self.hamiltonian = -1*self.hamiltonian
        print(self.hamiltonian)

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
                self.circuit.h(qub)
            # Because there are grouped parameters ... we have to do a lazy method
            for _ in range(1):
                self.new_param()
                for pair in self.connections:
                    self.circuit.rzz(2 * self.get_last_param(),list(self.qubits)[pair[0]],list(self.qubits)[pair[1]])
                self.new_param()
                for qub in list(self.qubits):
                    self.circuit.rx(2 * self.get_last_param(), qub)
                # set up parameter for p2
            self.circuit.measure_all()
            self.initialize_params()
            print(self.circuit)
        else:
            print("Model already initialized - Parameters exist.")

    def set_backend(self,backend):
        self.backend = backend

    def cost_function(self,statevector):
        print(np.matmul(np.matmul(statevector.T,self.hamiltonian),statevector))
        return np.matmul(np.matmul(statevector.T,self.hamiltonian),statevector)

