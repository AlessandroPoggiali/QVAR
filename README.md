# QVAR
Quantum subroutine for VARiance estimation

The QVAR quantum subroutine employs a gate-based circuit logarithmic in depth to compute the classical variance of a set of values stored in superposition. This subroutine uses the Amplitude Estimation algorithm to estimate the variance of the indexed values. 

## How to use it

The QVAR method in `qvar.py` is responsible for creating and executing the quantum circuit for computing the variance of a set of values stored in the quantum superposition. To create the initial superposition of values of which you want to compute the variance, QVAR accepts a parameter `U`, which represents the unitary for creating the state of interest, and a parameter `var_index` for indexing the target values in the superposition.

According to the parameter `version`, you can run the QVAR subroutine in the following ways:

* `AE`    : it will run the standard Amplitude Estimation algorithm with the related parameter `eval_qubits`. 
* `FAE`   : it will run the Faster Amplitude Estimation algorithm with the releted parameters `delta` and `max_iter`
* `SHOTS` : (only for debugging purposes) this version does not use Amplitude Estimation. Instead, it will estimate the variance using a number of repetitions of the quantum circuit given by the parameter `shots`. 

Additionally, you can specify a multiplicative constant used to obtain the final value through the `normalization_factor` parameter. Finally, flagging `postprocessing` parameter as True, the QVAR subroutine returns the posprocessed value obtained through the *Maximum Likelihood Estimator* technique.


## Basic example

To run a simple demostration of the QVAR subroutine, follow these steps:
* Make sure you have Qiskit installed on your computer
* Clone this repo with `git clone https://github.com/AlessandroPoggiali/QVAR.git`
* Navigate to the QVAR directory and run the command `python3 test.py`

The `test.py` file contains code that will run two demonstrations of the QVAR subroutine: the first one will compute the variance of the state vector of a random unitary, while the second one will compute the variance of a set of real values encoded through the FF-QRAM algorithm. The MSE with respect to the classical variance over 5 executions will appear on the terminal.

