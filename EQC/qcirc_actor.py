import ray
import qiskit
import numpy as np
import time
from qiskit.providers.ibmq.managed import IBMQJobManager
import datetime


# ========================================================================================
# This is the client code. Each client is initiated with this class, and holds all of its methods
# This is how the asynchronous updating and requesting of circuits is done.
# ========================================================================================
@ray.remote
class RemoteWorker:
    def __init__(self, backend, circuit, parameters,
                 index, cost_function):
        qiskit.IBMQ.load_account()
        self.bck, self.fwd = None, None
        try:
            # ========================================================================================
            # Change this to your provider
            # ========================================================================================
            provider = None
            self.backend = provider.get_backend(backend)
        except:
            try:
                provider = qiskit.IBMQ.get_provider(hub='ibm-q')
                self.backend = provider.get_backend(backend)
            except:
                self.backend = backend
        self.index = index
        self.cost_function = cost_function
        self.parameter_values = parameters
        self.circuit = circuit
        self.k = self.circuit.num_qubits
        self.generate_gradient_circuits()
        self.circuits = qiskit.transpile(self.circuits, backend=self.backend, optimization_level=3,
                                         coupling_map=self.backend.configuration().coupling_map)

    def update_circuit(self,circuit,parameters,index):
        self.index = index
        self.parameter_values = parameters
        self.circuit = circuit
        self.generate_gradient_circuits()
        self.circuits = qiskit.transpile(self.circuits, backend=self.backend, optimization_level=3,
                                         coupling_map=self.backend.configuration().coupling_map)

    def generate_gradient_circuits(self):
        self.differentiate_parameters()
        fwd = self.circuit.assign_parameters(self.fwd)
        bck = self.circuit.assign_parameters(self.bck)
        self.circuits = [fwd, bck]

    def differentiate_parameters(self):
        self.fwd = self.parameter_values.copy()
        # ========================================================================================
        # You can fine tune the coefficient to the differentiator. Things such as QAOA have extremely
        # sensitive cost landscapes, so reducing your finite difference methods bounds can improve performance.
        # Changing the 0.5 to a 0.05 for example, can greatly increase your convergence in QAOA examples.
        # ========================================================================================
        self.fwd[self.index] += 0.5*np.pi
        self.bck = self.parameter_values.copy()
        self.bck[self.index] -= 0.5*np.pi


    def complete_results(self,r_dict:dict):
        # ========================================================================================
        # QAPP might produce only n values, yet the feature space is 2^n
        # To tackle this, generate and append all binary maps that are not featured.
        # ========================================================================================
        n_bits = len(list(r_dict.keys())[0])
        b_str = ''.join(['1' for _ in range(n_bits)])
        for i in range(int(b_str,2)+1):
            key_b = bin(i).lstrip('0b').zfill(n_bits)
            if key_b not in r_dict:
                r_dict[key_b] = 0
        return r_dict

    def to_p_dist(self, r_dict: dict):
        # ========================================================================================
        # Convert a count-based dictionary to a probability-based dictionary
        # ========================================================================================
        r_dict = self.complete_results(r_dict)
        keys = list(r_dict.keys())
        keys.sort()
        results = np.array([r_dict[key] for key in keys])
        shots = sum(results)
        return results / shots

    def calculate_gate_gradients(self,fwd_state,bck_state):
        gradients = []
        for f_gate,b_gate in zip(fwd_state,bck_state):
            try:
                f_cost = self.cost_function(f_gate)
                b_cost = self.cost_function(b_gate)
                gradient = (f_cost-b_cost)/2
            except:
                print(f_gate)
                print(b_gate)
            gradients.append(gradient)
        gradients = np.array(gradients)
        return gradients


    def run_circuit(self):
        # ========================================================================================
        # If backend set to sim, this will not work , hence try except
        # ========================================================================================
        try:
            job_manager = IBMQJobManager()
            job_set_foo = job_manager.run(self.circuits, backend=self.backend,shots=8192)
            start = time.time()
            while True:
                status = job_set_foo.statuses()[0] == job_set_foo.statuses()[0].DONE
                if status:
                    break
                else:
                    run_time = time.time() - start
                    # 1 hor is the limit
                    if run_time > 10400:
                        status = False
                        job_set_foo.cancel()
                        print("Run time exceeded 1 hour - breaking")
                        break
            if status:
                # GET RESULTS
                print("Succesful")
                for i in range(2):
                    print(f"RESULTS {i}: {job_set_foo.results().get_counts(i)}")
            else:
                print(f"Failed to run on {self.backend}")
                # No gradient, No index
                return self.index, np.array([0.0]), 0.0
            results = [job_set_foo.results().get_counts(i) for i in range(len(self.circuits))]
        except:
            job = self.backend.run(self.circuits,shots=8192)
            results = job.result().get_counts()
        dist_list = []
        for index,circuit in enumerate(self.circuits):
            result = results[index]
            result = self.complete_results(result)
            dist_list.append(self.to_p_dist(result))
        dist_list = np.array(dist_list)
        fwd_vector = dist_list[:int(len(dist_list)/2)]
        bck_vector = dist_list[int(len(dist_list)/2):]
        grad = self.calculate_gate_gradients(fwd_vector,bck_vector)
        try:
            p_error = self.calculate_p_err(self.circuit,self.backend,self.circuit.num_qubits)
        except:
            p_error = 1
        print(p_error)
        return self.index, grad, p_error

    def get_circuit_cost(self):
        results = self.backend.run(self.circuit,shots=8000)
        result = results.result().get_counts()
        result = self.complete_results(result)
        return self.cost_function(self.to_p_dist(result))

    def calculate_p_err(self,circuit, backend, k):
        # ========================================================================================
        # P error calculation model from paper. This is assuming the implementation of
        # backend.properties().to_dict() has been sufficiently implemented.
        # ========================================================================================

        sq_names = ['ID', 'RZ', 'SX', 'X']
        cx_names = ['CX']
        RESULTS = {}
        t_circs = qiskit.transpile(circuit, backend=backend)
        t_sq = 0
        t_cx = 0
        for key in t_circs.count_ops().keys():
            if key.upper() in sq_names:
                t_sq += t_circs.count_ops()[key]
            elif key.upper() in cx_names:
                t_cx += t_circs.count_ops()[key]
            else:
                pass
        data = backend.properties().to_dict()
        now = datetime.datetime.now()
        t_calibration = datetime.datetime.timestamp(now) - datetime.datetime.timestamp(data['last_update_date'])
        # Collects list of CNOT gate time and SQG gate time
        cx_times = []
        sq_times = []
        cx_error = []
        sq_error = []
        for gate_info in data['gates']:
            if len(gate_info["parameters"]) < 2:
                continue
            if gate_info["gate"] == "cx":
                cx_times.append(gate_info["parameters"][1]["value"])
                cx_error.append(gate_info["parameters"][0]["value"])
            else:
                time = gate_info["parameters"][1]["value"]
                sq_times.append(time)
                sq_error.append(gate_info["parameters"][0]["value"])
        t1_times = []
        m_error = []
        t2_times = []
        for qubit in data['qubits']:
            t1_times.append(qubit[0]['value'])
            t2_times.append(qubit[1]['value'])
            m_error.append(qubit[4]['value'])
        t1_times = np.array(t1_times)
        t2_times = np.array(t2_times)
        sq_times = np.array(sq_times)
        cx_times = np.array(cx_times)
        m_error = np.array(m_error)
        cx_error = np.array(cx_error)
        sq_error = np.array(sq_error)
        t1 = np.mean(t1_times) * 1000  # Unit conversion
        t2 = np.mean(t2_times) * 1000  # Unit conversion
        sq = np.mean(sq_error)
        cx = np.mean(cx_error)
        m = np.mean(m_error)
        transpiled_circ = qiskit.transpile(circuit, backend=backend)
        depth = transpiled_circ.depth()
        t_mu = (cx_times.mean() + sq_times.mean()) / 2
        return self.get_p(t1, t2, t_mu, depth, sq, t_sq, cx, t_cx, m, k)

    def get_p(self,t1,t2,t_mu,cd,g1_error,g1,g2_error,g2,m_error,m):
        return np.exp((-1*t_mu*cd)/(t1*t2))*(1-g1_error)**g1 * (1-g2_error)**g2 * (1-m_error)**m