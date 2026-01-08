import numpy as np
import math
import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_algorithms import EstimationProblem

from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import *
from qiskit_ibm_runtime import SamplerV2
from qiskit.providers.fake_provider import *
from qiskit_aer import AerSimulator

from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix
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

def QVAR(
    U,
    var_index=None,
    ps_index=None,
    n_h_gates=0,
    postprocessing=True,
    backend=None,
):

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

    # Objective qubits (kept for compatibility with AE/FAE)
    if ps_index is None:
        objective_qubits = list(range(1 + e_qbits))
    else:
        objective_qubits = (
            list(range(1 + e_qbits))
            + [qc.num_qubits - u_qbits + x for x in ps_index]
        )

    EstimationProblem(
        state_preparation=qc,
        objective_qubits=objective_qubits,
    )

    # Default backend
    if backend is None:
        backend = AerSimulator(method="statevector")
    #return transpile(qc, backend)
    return _run_and_extract_state(qc, backend)

def QVAR_old(
    U,
    var_index=None,
    ps_index=None,
    n_h_gates=0,
    postprocessing=True,
    backend=None,
):

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

    if ps_index is None:
        objective_qubits = list(range(1 + e_qbits + q_qbits))
    else:
        objective_qubits = (
            list(range(1 + e_qbits + q_qbits))
            + [qc.num_qubits - u_qbits + x for x in ps_index]
        )

    EstimationProblem(
        state_preparation=qc,
        objective_qubits=objective_qubits,
    )

    if backend is None:
        backend = AerSimulator(method="statevector")
    #return transpile(qc, backend)
    return _run_and_extract_state(qc, backend)


def test_noise():

    N_list = [16]
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

                sv_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], n_h_gates=n, backend=AerSimulator.from_backend(GenericBackendV2(3*n+2, noise_info=False, seed=s*123), noise_model=None))
                sv_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], n_h_gates=n, backend=AerSimulator.from_backend(GenericBackendV2(2*n+2, noise_info=False, seed=s*123), noise_model=None))
                dm_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], n_h_gates=n, backend=AerSimulator.from_backend(GenericBackendV2(3*n+2, noise_info=True, seed=s*123)))
                dm_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1],  n_h_gates=n, backend=AerSimulator.from_backend(GenericBackendV2(2*n+2, noise_info=True, seed=s*123)))

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
        
    '''
    plt.scatter(N_list, mean_trials_old, label="Fidelity old")
    plt.scatter(N_list, mean_trials_new, label="Fidelity new")
    plt.xticks(N_list)
    plt.legend()
    plt.savefig("fidelity.png")
    '''

    rows = zip(N_list, mean_trials_old, std_trials_old, mean_trials_new, std_trials_new)

    # Write to a CSV file
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N', 'mean_fidelity_old', 'std_fidelity_old', 'mean_fidelity_new', 'std_fidelity_new'])  # Write header
        writer.writerows(rows)  # Write rows
 



'''

def choose_fake_backend(n_qubits):
    fakes = [FakeManilaV2(), FakeMelbourneV2()]
    for b in fakes:
        if b.num_qubits >= n_qubits:
            return b
    raise ValueError("No fake backend large enough")

def probability_distribution_from_sampler(qc, sampler, shots=5000):
    # Ensure there is a classical register measuring all qubits
    if qc.num_clbits == 0:
        creg = ClassicalRegister(qc.num_qubits, name="cr")
        qc.add_register(creg)
        qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    
    result = sampler.run([qc], shots=shots).result()
    pub = result[0]  # first circuit's result
    counts = pub.data.cr.get_counts()
    # Normalize to probabilities
    probs = {b: c/shots for b, c in counts.items()}
    return probs




def classical_fidelity(p_noisy, p_ideal):
    """Compute fidelity proxy: F = (sum_sqrt(sqrt(p_i * p_n)))^2"""
    # All bitstrings in union
    keys = set(p_ideal.keys()).union(p_noisy.keys())
    f = sum(np.sqrt(p_ideal.get(k,0) * p_noisy.get(k,0)) for k in keys)
    return f**2

def test_old_vs_new(shots=5000, trials=5):
    N_list = [2, 4, 8]

    mean_old, std_old = [], []
    mean_new, std_new = [], []

    rng = np.random.default_rng(seed=123)

    for N in N_list:
        n = math.ceil(math.log2(N))
        total_qubits = 3*n + 2  # QVAR_old qubit count

        # Choose smallest backend that fits
        backend_hw = choose_fake_backend(total_qubits)
        noisy_backend = AerSimulator.from_backend(backend_hw)
        sampler = SamplerV2(noisy_backend)

        print(f"\nN={N}, qubits={total_qubits}, backend={backend_hw.name}")

        fidelities_old = []
        fidelities_new = []

        for _ in range(trials):
            vector = rng.uniform(-1, 1, N)

            r = QuantumRegister(1, 'r')
            i = QuantumRegister(n, 'i')
            U = QuantumCircuit(i, r)

            # Example state preparation (replace with your _register_switcher + mcry)
            U.h(i)
            for index, val in enumerate(vector):
                _register_switcher(U, index, i)
                U.mcry(2*np.arcsin(val), i[:], r[0])
                _register_switcher(U, index, i)

            # --- Ideal statevector for fidelity ---
            sv_old = Statevector(QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1],
                                         backend=AerSimulator(method="statevector")))
            sv_new = Statevector(QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1],
                                      backend=AerSimulator(method="statevector")))

            # Convert ideal statevector to probability distribution
            probs_old_ideal = sv_old.probabilities_dict()
            probs_new_ideal = sv_new.probabilities_dict()

            # --- Noisy execution with SamplerV2 ---
            probs_old_noisy = probability_distribution_from_sampler(QVAR_old(U, var_index=list(range(n)),
                                                                             ps_index=[U.num_qubits-1],
                                                                             backend=noisy_backend),
                                                                   sampler, shots=shots)
            probs_new_noisy = probability_distribution_from_sampler(QVAR(U, var_index=list(range(n)),
                                                                          ps_index=[U.num_qubits-1],
                                                                          backend=noisy_backend),
                                                                    sampler, shots=shots)

            # Fidelity proxies
            f_old = classical_fidelity(probs_old_noisy, probs_old_ideal)
            f_new = classical_fidelity(probs_new_noisy, probs_new_ideal)

            fidelities_old.append(f_old)
            fidelities_new.append(f_new)

        mean_old.append(np.mean(fidelities_old))
        std_old.append(np.std(fidelities_old))
        mean_new.append(np.mean(fidelities_new))
        std_new.append(np.std(fidelities_new))

    # Save results
    rows = zip(N_list, mean_old, std_old, mean_new, std_new)
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "mean_fidelity_old", "std_old",
                         "mean_fidelity_new", "std_new"])
        writer.writerows(rows)





'''
def plot_fidelity_results(csv_file="results.csv"):
    N_list = []
    mean_old = []
    std_old = []
    mean_new = []
    std_new = []

    # Read CSV
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            N_list.append(int(row["N"]))
            mean_old.append(float(row["mean_fidelity_old"]))
            std_old.append(float(row["std_fidelity_old"]))
            mean_new.append(float(row["mean_fidelity_new"]))
            std_new.append(float(row["std_fidelity_new"]))

    # Plot
    plt.figure(figsize=(7,5))
    plt.errorbar(N_list, mean_old, yerr=std_old, fmt='o-', label="QVAR_old", capsize=4)
    plt.errorbar(N_list, mean_new, yerr=std_new, fmt='s-', label="QVAR", capsize=4)

    plt.xlabel("Input dimension N")
    plt.ylabel("Fidelity (noisy vs ideal)")
    plt.title("Noise resilience comparison: QVAR vs QVAR_old")
    plt.xticks(N_list)
    plt.ylim(0,1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()







def print_result():
    df = pd.read_csv('results.csv')
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
    plt.savefig("fidelity.png")

if __name__ == "__main__":

    
    '''
    service = QiskitRuntimeService()

    for b in service.backends(simulator=False, operational=True):
        print(b.name)

    backend_hw = service.backend("ibm_fez")
    backend = AerSimulator.from_backend(backend_hw)
    backend = AerSimulator(method="statevector")


    backend = service.backend("ibm_torino")
    '''

    test_noise()
    plot_fidelity_results("results.csv")