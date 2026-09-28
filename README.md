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

## Citation

If you use or adapt the code in this repository, please cite the following works:

1. **Original version — ESANN 2023**

```bibtex
@inproceedings{poggiali2023quantum,
  title={Quantum Feature Selection with Variance Estimation.},
  author={Poggiali, Alessandro and Bernasconi, Anna and Berti, Alessandro and Del Corso, Gianna M and Guidotti, Riccardo and others},
  booktitle={ESANN},
  year={2023}
}
```

2. **Journal version — Quantum Machine Intelligence, 2024**

```bibtex
@article{bernasconi2024quantum,
  title={Quantum subroutine for variance estimation: algorithmic design and applications},
  author={Bernasconi, Anna and Berti, Alessandro and Del Corso, Gianna M and Guidotti, Riccardo and Poggiali, Alessandro},
  journal={Quantum Machine Intelligence},
  volume={6},
  number={2},
  pages={78},
  year={2024},
  publisher={Springer}
}
```

3. **Current circuit implementation / improved version — Quantum Machine Intelligence, 2026**

```bibtex
@article{poggiali2026more,
  title={A more efficient quantum circuit for estimating the variance},
  author={Poggiali, Alessandro and Ju, Jiwon},
  journal={Quantum Machine Intelligence},
  volume={8},
  number={1},
  pages={34},
  year={2026},
  publisher={Springer}
}
```

The current implementation of the variance-estimation circuit corresponds to the improved circuit described in Poggiali and Ju (2026). The earlier versions and the development of the method are described in Bernasconi et al. (2024) and Poggiali et al. (2023).

The code in this repository is released under the MIT License. See the `LICENSE` file for the full license text.

When reusing or adapting code from this repository, please retain the copyright and license notices and cite the relevant publications above.
