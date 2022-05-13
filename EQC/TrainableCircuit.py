import numpy as np
import qiskit
import sys
import random
import matplotlib.pyplot as plt
import os
import json
from qiskit.circuit import Parameter
from QuantumCircuitOptimizer import QuantumCircuitOptimizer


class TrainableCircuit():
    def __init__(self,k,entanglement='full',seed=42):
        self.PAULI_MATRICES = {'Z':np.array([[1, 0], [0, -1]]),
                               'X':np.array([[0,1],[1,0]]),
                               'Y':np.array([[0,0-1j],[0+1j,0]]),
                               'I': np.array([[1,0],[0,1]])}
        self.parameters = []
        self.k = k
        self.parameter_values = None
        self.circuit = None
        self.backend = None

    def initialize_params(self):
        if len(self.parameters) == 0:
            print("Your circuit is empty with no trainable parameters")
            return
        else:
            print(f"Initializing {len(self.parameters)} parameter values...")
            np.random.seed(self.seed)
            self.parameter_values = np.random.rand(len(self.parameters))*np.pi

    def new_param(self):
        n_p = Parameter(f"theta_{len(self.parameters)}")
        self.parameters.append(n_p)
        return n_p

    def get_last_param(self):
        return self.parameters[-1]

    def train(self,epochs=100,learning_rate=0.01,
              generate_evaluation_metrics = False):
        if self.parameters is None or len(self.parameters) == 0:
            print("Can not train - Model not initialized - please call model.initialize")
            return
        else:
            optimizer = QuantumCircuitOptimizer(self.circuit,self.parameters,self.parameter_values,
                                                self.cost_function,learning_rate,
                                                self.backend,epochs)
            optimizer.assign_cost_function(self.cost_function)
            optimizer.train()
