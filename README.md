
# EQC - Ensembled Quantum Computing for Variational Quantum Algorithms (ISCA'22)

The following repository contains the software used to implement EQC, a distributed variational quantum algorithm optimizer. Please look at our ISCA-22 paper (DOI: 10.1145/3470496.3527434) see below for details.

EQC is a single-master node multi-client multi-quantum-processor system that utilized asynchronous stochastic gradient descent to optimize variational quantum algorithms. 
Key benefits of EQC include substantial reduction in training time, mitigation of device-specific bias and time-dependant drift, and the ability to homogenise multiple quantum providers to one virtualized quantum backend.

# Prerequisites

The core requirements for EQC include using Python 3.7.0+, and Ray.io. The dependancies 
of your system will also depend on the quantum provider you are connecting to. For 
usage with IBM Quantum Experience, using Qiskit would be recommended. However, 
providers such as IonQ, Rigetti, etc, will require installation and usage of their
required packages.  

# Utilizing EQC 

To use EQC, generating a similar-styled object that inherits from the TrainableCircuit
class. This requires the declaration

```python
class YourVariationalAlgorithm(TrainableCircuit):
```
Once your class is implemented, it should contain sufficient information similar to how 
QAOA or VQE are implemnted in this repot. These classes include methods such as 
cost function, which defines the objects optimization function.

Having implented your variational algorithm, the train function is called.
This creates the QuantumCircuitOptimizer object, which is the master node of the system. 
This object keeps track of global parameters, which iteration it is on, which parameter is next etc.
The train function is based off of the Algorithm 1 presented in the ISCA paper.

The QuantumCircuitOptimizer initializes a set of client nodes, based on Algorithm 2.
The RemoteWorker object represents a client node, and handles one-on-one communication with
the assigned quantum processor.

## Citation format

If you find this repository useful, please cite our ISCA-22 paper:
 - Samuel Stein, Nathan Wiebe, Yufei Ding, Bo Peng, Karol Kowalski, Nathan Baker, James Ang, and Ang Li. "EQC: Ensembled Quantum Computing for Variational Quantum Algorithms" In Proceedings of the International Symposium on Computer Architecture, ACM, 2022.

Bibtex:
```text
@inproceedings{li2020density,
    title={EQC: Ensembled Quantum Computing for Variational Quantum Algorithms},
    author={Stein, Samuel and Wiebe, Nathan and Ding, Yufei and Peng, Bo and Kowalski, Karol and Baker, Nathan and Ang, James and Li, Ang},
    booktitle={Proceedings of the International Symposium on Computer Architecture},
    year={2022}
}
``` 




## License

This project is licensed under the BSD License, see [LICENSE](LICENSE) file for details.




# Acknowledgements
**PNNL-IPID: 32474-E**

This material is based upon work supported by the U.S. Department of Energy, Office of Science, National Quantum Information Science Research Centers, Co-design Center for Quantum Advantage (C2QA) under contract number DE-SC0012704. The Pacific Northwest National Laboratory is operated by Battelle for the U.S. Department of Energy under contract DE-AC05-76RL01830.
