import numpy as np
import qiskit
import sys
import random
import matplotlib.pyplot as plt
import os
import json
from Optimizer import Adam
from qcirc_actor import RemoteWorker
import ray
import time
import pathlib
from Optimizer import Adam,SGD

class QuantumCircuitOptimizer:
    def __init__(self,circuit,parameters,parameter_values,cost_function,learning_rate,backends,epochs=100):
        self.circuit = circuit
        self.optimizer = SGD(learning_rate=0.1)
        self.parameters = parameters
        self.parameter_values = parameter_values
        self.epochs = epochs
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.backends = backends
        print(self.backends)

    def assign_cost_function(self,cost_function):
        self.cost_function = cost_function

    def update_params(self,gradient,epoch,index,learning_rate=0.01,descent=True):
        if descent is True:
            z = -1
        else:
            z = 1
        new_values = []
        print(f"Old value: {self.parameter_values[index]}\nGradient:{gradient}\n")
        self.parameter_values[index] = self.optimizer.update(epoch,self.parameter_values[index],gradient,z)
        print(f"New Value: {self.parameter_values[index]}")

    def complete_results(self,r_dict:dict):
        # QAPP might produce only n values, yet the feature space is 2^n
        # To tackle this, generate and append all binary maps that are not featured.
        n_bits = len(list(r_dict.keys())[0])
        b_str = ''.join(['1' for _ in range(n_bits)])
        for i in range(int(b_str,2)+1):
            key_b = bin(i).lstrip('0b').zfill(n_bits)
            if key_b not in r_dict:
                r_dict[key_b] = 0
        return r_dict

    def get_cost(self):
        circuit = self.circuit.assign_parameters(self.parameter_values)
        sim = qiskit.Aer.get_backend('aer_simulator')
        circ = qiskit.transpile(circuit,sim)
        result = sim.run(circ).result().get_counts()
        result = self.complete_results(result)
        keys = list(result.keys())
        keys.sort()
        results = np.array([result[key] for key in keys])
        shots = sum(results)
        wavefunc = results/shots
        return self.cost_function(wavefunc)

    def save_info(self,file_path,file):
        # ========================================================================================
        # Function to save information to a results folder. Must exist prior.
        # This can be changed to pickle.dump(), however pickle.dump() does not play well
        # with some Backend objects from Qiskit. Hence using a simple string write.
        # ========================================================================================
        path = str(pathlib.Path(__file__).parent.resolve())
        path += "/results"
        print(path)
        with open(path+"FILE_PATH",'w') as f:
            f.write(str(file))

    def train(self,epochs=250,learning_rate=0.,validate=True):
        curr_epoch = 0
        # ========================================================================================
        # These variables take care of tracking system performance. This is used in the generation
        # of analytics for performance
        # ========================================================================================
        self.parameter_history = []
        self.workers = []
        self.loss = []
        index = 0
        n_complete = 0
        ep_pr_hr = []
        # ========================================================================================
        # Cold start max and min values set, updated after 1st returned value from any machine.
        # ========================================================================================
        max_p_correct = 0.7
        min_p_correct = 0.5
        start_epoch = time.time()
        for backend in self.backends:
            wrkr = RemoteWorker.remote(backend, self.circuit, self.parameter_values, index,
                                       self.cost_function)
            self.workers.append(wrkr)
            index +=1
            if index >= len(self.parameter_values):
                index = 0
                curr_epoch +=1
        futures = [c.run_circuit.remote() for c in self.workers]
        while True:
            if time.time()-start_epoch > 15*60:
                ep_pr_hr.append(n_complete*4)
                start_epoch = time.time()
                n_complete = 0
            ready, not_ready = ray.wait(futures)
            print(f"Ready: {ready}")
            for obj in ready:
                diff_index,grad,p_correct = ray.get(obj)
                # ========================================================================================
                # For the weighting system, we set the bound here. This wil be used in the normalisation
                # calculation later. If this is not desired, simply set min and max to 0 and 1 respectively.
                # If we find a machine with a better or worse performance than observed, we update the new
                # max or min p_correct to that.
                # ========================================================================================
                if p_correct > max_p_correct:
                    max_p_correct = p_correct
                    print(f"Updated max_p_c to {p_correct}")
                if p_correct < min_p_correct:
                    min_p_correct = min_p_correct
                # ========================================================================================
                # Use the below equation, adapted accordingly, if you want to use a weighted gradient descent
                # approach. However, if you want to use unweighted, comment out the below line and leave it as
                # grad = grad. NOTE: Modify the below equation accordingly to scale your weights according
                # to the paper
                # ========================================================================================
                grad = grad * (0.25*((p_correct - min_p_correct)/(max_p_correct - min_p_correct))+0.75)
                #grad = grad
                # ========================================================================================
                # This print section prints out information of the returned values from each cilent node
                # when it is returned. Comment out if not desired.
                # ========================================================================================
                print("Returned Values:")
                print(diff_index)
                print(grad)
                print(p_correct)
                print('-'*20)

                # ========================================================================================
                # Logic behind distributing the optimization accross multiple nodes below.
                # Mimic of Algorithm 1 in EQC paper
                # ========================================================================================
                self.update_params(grad,curr_epoch,diff_index,learning_rate=learning_rate,descent=True)
                for indx, future in enumerate(futures):
                    if future == obj:
                        r_worker = indx
                index += 1
                old_epoch = curr_epoch
                if index >= len(self.parameter_values):
                    index = 0
                    curr_epoch += 1
                print(f"Updating Working {r_worker} with Index {index}")
                self.workers[r_worker].update_circuit.remote(self.circuit, self.parameter_values, index)
                futures[r_worker] = self.workers[r_worker].run_circuit.remote()
                if curr_epoch != old_epoch:
                    n_complete += 1  # For evaluating epochs/hr
                    cost = self.get_cost()
                    self.loss.append(cost)
                    self.parameter_history.append(list(self.parameter_values))
                    t_params = self.parameter_history.copy()
                    t_params = list(t_params)
                    # ========================================================================================
                    # Enable this section of code if you want to save performance, costs, backend list, and parameters
                    # history to the save_info function mentioned above.
                    # ========================================================================================
                    # self.save_info(None,{"costs": self.loss,
                    #                     "performance":ep_pr_hr,"backends":self.backends,
                    #                      "parameters":t_params})
                    print(f" Current Cost: {cost}")
                    print(self.parameter_values)

                print("Running a new circuit")
            print(f"Not Ready: {not_ready}")
            time.sleep(0.01)
            if curr_epoch == epochs:
                break
