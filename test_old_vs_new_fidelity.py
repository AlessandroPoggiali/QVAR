import numpy as np
import math
import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem


from qiskit_ibm_runtime.fake_provider import *
from qiskit.providers.fake_provider import *
from qiskit_aer import AerSimulator

from qiskit.quantum_info import Statevector, state_fidelity
import matplotlib.pyplot as plt
import csv
import pandas as pd

def _register_switcher(circuit, value, qubit_index):
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1]
    for idx, bit in enumerate(value):
        if not int(bit):
            circuit.x(qubit_index[idx])

def _run_and_extract_state(qc, backend):
    """
    Runs qc on a Qiskit 2.0 backend and returns either
    a statevector or density matrix, depending on noise.
    """

    qc_t = transpile(qc, backend)
   
    # Noisy backend → density matrix
    if isinstance(backend, AerSimulator) and backend.options.noise_model is not None:
        qc_t.save_density_matrix()
        result = backend.run(qc_t).result()
        return np.asarray(result.data(0)["density_matrix"])

    # Ideal backend → statevector
    qc_t.save_statevector()
    result = backend.run(qc_t).result()
    return np.asarray(result.data(0)["statevector"])

def test_QVAR_fidelity_old(U, var_index=None, ps_index=None, backend=None):

    if var_index is None:
        var_index = list(range(U.num_qubits))

    i_qbits = len(var_index)
    e_qbits = q_qbits = i_qbits
    u_qbits = U.num_qubits

    a = QuantumRegister(1, "a")
    e = QuantumRegister(e_qbits, "e")
    q = QuantumRegister(q_qbits, "q")
    u = QuantumRegister(u_qbits, "u")

    qc = QuantumCircuit(a, e, q, u)

    # --- State preparation ---
    st_ff = Statevector.from_instruction(U)
    qc.append(StatePreparation(st_ff.data), u[:])

    qc.h(a)
    qc.cx(a, e)
    qc.x(e)

    for t in range(i_qbits):
        qc.cswap(a, q[t], u[var_index[t]])

    for t in range(i_qbits):
        qc.ch(a, u[var_index[t]])

    qc.ch(a, e)
    qc.h(a)

    qc.h(q)
    qc.x(q)

    if backend is None:
        backend = AerSimulator(method="statevector")

    return _run_and_extract_state(qc, backend)


def test_QVAR_fidelity_new(U, var_index=None, backend=None):

    if var_index is None:
        var_index = list(range(U.num_qubits))

    i_qbits = len(var_index)
    e_qbits = i_qbits
    u_qbits = U.num_qubits

    a = QuantumRegister(1, "a")
    e = QuantumRegister(e_qbits, "e")
    u = QuantumRegister(u_qbits, "u")

    qc = QuantumCircuit(a, e, u)

    # --- State preparation ---
    st_ff = Statevector.from_instruction(U)
    qc.append(StatePreparation(st_ff.data), u[:])

    qc.h(a)

    for t in range(i_qbits):
        qc.cswap(a, e[t], u[var_index[t]])

    qc.ch(a, e)
    for t in range(i_qbits):
        qc.ch(a, u[var_index[t]])

    qc.x(e)
    qc.h(a)

    # Default backend
    if backend is None:
        backend = AerSimulator(method="statevector")

    return _run_and_extract_state(qc, backend)



def test_fidelity():

    N_list = [2,4,8,16]
    trial = 5
    seeds = 3

    mean_trials_old = []
    mean_trials_new = []
    std_trials_old = []
    std_trials_new = []

    for N in N_list:
        print("N = " + str(N))
        cl = []
        q_old = []
        q_new = []
        
        n = math.ceil(math.log2(N))
        
        for t in range(trial):
            np.random.seed(123*t)
            vector = np.random.uniform(-1,1, N)
            classical = np.var(vector)
            cl.append(classical)
            
            q_old_seed = []
            q_new_seed = []
            for s in range(seeds):

                r = QuantumRegister(1, 'r') 
                i = QuantumRegister(n, 'i')  
            
                U = QuantumCircuit(i, r)
                
                U.h(i)
                
                for index, val in zip(range(len(vector)), vector):
                    _register_switcher(U, index, i)
                    U.mcry(np.arcsin(val)*2, i[0:], r) 
                    _register_switcher(U, index, i)

                sv_old = test_QVAR_fidelity_old(U, var_index=list(range(n)), backend=AerSimulator.from_backend(GenericBackendV2(3*n+2, noise_info=False, seed=s*123), noise_model=None))
                sv_new = test_QVAR_fidelity_new(U, var_index=list(range(n)), backend=AerSimulator.from_backend(GenericBackendV2(2*n+2, noise_info=False, seed=s*123), noise_model=None))
                dm_old = test_QVAR_fidelity_old(U, var_index=list(range(n)), backend=AerSimulator.from_backend(GenericBackendV2(3*n+2, noise_info=True, seed=s*123)))
                dm_new = test_QVAR_fidelity_new(U, var_index=list(range(n)), backend=AerSimulator.from_backend(GenericBackendV2(2*n+2, noise_info=True, seed=s*123)))

                #print("statevector\n"+str(sv_old))
                #print("density matrix\n"+str(dm_old))
                
                old_fidelity = state_fidelity(sv_old, dm_old)
                new_fidelity = state_fidelity(sv_new, dm_new)
                

                print("fidelity old: " + str(old_fidelity))
                print("fidelity new: " + str(new_fidelity))

                q_old_seed.append(old_fidelity)
                q_new_seed.append(new_fidelity)
                
            q_old.append(np.mean(q_old_seed))
            q_new.append(np.mean(q_new_seed))

        
        mean_trials_old.append(np.mean(q_old))
        mean_trials_new.append(np.mean(q_new))
        std_trials_old.append(np.std(q_old))
        std_trials_new.append(np.std(q_new))


    rows = zip(N_list, mean_trials_old, std_trials_old, mean_trials_new, std_trials_new)

    # Write to a CSV file
    with open('results/fidelity_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N', 'mean_fidelity_old', 'std_fidelity_old', 'mean_fidelity_new', 'std_fidelity_new'])  # Write header
        writer.writerows(rows)  # Write rows
 

def print_result():
    df = pd.read_csv('results/fidelity_results.csv')
    N_list = df['N']
    mean_fidelity_old = df['mean_fidelity_old']
    mean_fidelity_new = df['mean_fidelity_new']
    std_fidelity_old = df['std_fidelity_old']
    std_fidelity_new = df['std_fidelity_new']
    fig, ax = plt.subplots(figsize=(10,8))

    #color1 = '#f1a340'
    #color2 = '#998ec3'
    color1 = '#fc8d59'
    color2 = '#91bfdb'
    a = 0.3

    plt.rcParams.update({'font.size': 22})
    plt.tick_params(labelsize=22)

    mean_fidelity_old = np.array(mean_fidelity_old)
    mean_fidelity_new = np.array(mean_fidelity_new)
    std_fidelity_old = np.array(std_fidelity_old)
    std_fidelity_new = np.array(std_fidelity_new)

    x = [str(n) for n in N_list]

    ax.errorbar(x, mean_fidelity_old, yerr=std_fidelity_old, label='QVAR old', color=color1, ecolor=color1, capsize=10, capthick=2)
    line_1, = plt.plot(x, mean_fidelity_old, color=color1, label='QVAR old')
    fill_1 = plt.fill_between(x, mean_fidelity_old - std_fidelity_old, mean_fidelity_old + std_fidelity_old, color=color1, alpha=a)

    ax.errorbar(x, mean_fidelity_new, yerr=std_fidelity_new, label='QVAR new', color=color2, ecolor=color2, capsize=10, capthick=2)
    line_2, = plt.plot(x, mean_fidelity_new, color=color2, label='QVAR new')
    fill_2 = plt.fill_between(x, mean_fidelity_new - std_fidelity_new, mean_fidelity_new + std_fidelity_new, color=color2, alpha=a)

    plt.xlabel('N', fontsize=22)
    plt.ylabel('Fidelity', fontsize=22)

    plt.legend([(line_1, fill_1), (line_2, fill_2)], ['QVAR old', 'QVAR new'])
    plt.savefig("plots/fidelity.png")

if __name__ == "__main__":

    test_fidelity()
    print_result()