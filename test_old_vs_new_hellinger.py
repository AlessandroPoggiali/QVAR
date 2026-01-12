from qiskit import *
from qiskit.circuit.library import *
from qiskit.circuit import *
from qiskit.quantum_info import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import numpy as np
from qiskit_aer import AerSimulator
import math
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import *
from qiskit.quantum_info.analysis import hellinger_distance
from qiskit_ibm_runtime import QiskitRuntimeService
import matplotlib.pyplot as plt
import csv
import pandas as pd

from qiskit.quantum_info import *
import matplotlib.pyplot as plt

from qiskit_algorithms import EstimationProblem
from qiskit_algorithms import AmplitudeEstimation, FasterAmplitudeEstimation
from qiskit_ibm_runtime import Sampler
from qiskit.quantum_info import Statevector

def test_QVAR_counts_old(U, var_index=None, ps_index=None, shots=8192, backend=None):

    if var_index is None:
        var_index = [x for x in range(U.num_qubits)]
    
    i_qbits = len(var_index)
    q_qbits = e_qbits = i_qbits
    u_qbits = U.num_qubits

    a = QuantumRegister(1,'a')
    e = QuantumRegister(e_qbits,'e')
    q = QuantumRegister(q_qbits,'q')
    u = QuantumRegister(u_qbits, 'u')

    ca = ClassicalRegister(1,'ca')
    cq = ClassicalRegister(q_qbits,'cq')
    ce = ClassicalRegister(e_qbits,'ce')
    if ps_index is not None:
        cps = ClassicalRegister(len(ps_index), 'cps')
        qc = QuantumCircuit(a, e, q, u, ca, cq, ce, cps)
    else:
        qc = QuantumCircuit(a, e, q, u, ca, cq, ce)
  
    st_ff = Statevector.from_instruction(U)
    qc.append(StatePreparation(st_ff.data), list(range(1+e_qbits+q_qbits, qc.num_qubits)))

    qc.h(a)
    qc.cx(a,e)
    qc.x(e)

    for t in range(i_qbits):
        qc.cswap(a,q[t],u[var_index[t]])

    for t in range(i_qbits):
        qc.ch(a,u[var_index[t]])
        
    qc.ch(a,e)
    qc.h(a)
    qc.h(q)
    qc.x(q)
    
    qc.measure(a, ca) 
    qc.measure(q, cq)
    qc.measure(e, ce)
    
    qc_t = transpile(qc, backend)    
    counts = backend.run(qc_t, shots=shots).result().get_counts()
    
    #print("depth old: " + str(qc_t.depth()))
    #print("size_old: " + str(qc_t.size()))
    #print("gate_fidelity_variance_old: " + str(backend._noise_info.gate_fidelity_variance))
    return counts
        

def test_QVAR_counts_new(U, var_index=None, ps_index=None, shots=8192, backend=None):

    if var_index is None:
        var_index = [x for x in range(U.num_qubits)]
    
    i_qbits = len(var_index)
    e_qbits = i_qbits
    u_qbits = U.num_qubits

    a = QuantumRegister(1,'a')
    e = QuantumRegister(e_qbits,'e')
    u = QuantumRegister(u_qbits, 'u')

    ca = ClassicalRegister(1,'ca')
    ce = ClassicalRegister(e_qbits,'ce')
    if ps_index is not None:
        cps = ClassicalRegister(len(ps_index), 'cps')
        qc = QuantumCircuit(a, e, u, ca, ce, cps)
    else:
        qc = QuantumCircuit(a, e, u, ca, ce)
         
    st_ff = Statevector.from_instruction(U)
    qc.append(StatePreparation(st_ff.data), list(range(1+e_qbits, qc.num_qubits)))
    qc.h(a)

    for t in range(i_qbits):
        qc.cswap(a,e[t],u[var_index[t]])

    qc.ch(a,e)
    for t in range(i_qbits):
        qc.ch(a,u[var_index[t]])
        
    qc.x(e)    
    qc.h(a)

    qc.measure(a, ca) 
    qc.measure(e, ce)
    
    qc_t = transpile(qc, backend)    
    counts = backend.run(qc_t, shots=shots).result().get_counts()

    #print("depth_new: " + str(qc_t.depth()))
    #print("size_new: " + str(qc_t.size()))
    #print("gate_fidelity_variance_new: " + str(backend._noise_info.gate_fidelity_variance))
    return counts
       
def _register_switcher(circuit, value, qubit_index):
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1]
    for idx, bit in enumerate(value):
        if not int(bit):
            circuit.x(qubit_index[idx])


def test_uniform():
    N_list = [2,4,8]
    trial = 10
    seeds = 1
    shots_list = [100 * x for x in N_list]

    mean_trials_old = []
    mean_trials_new = []
    std_trials_old = []
    std_trials_new = []

    service = QiskitRuntimeService()
    noisy_backend = service.backend("ibm_torino")
    noise_model = NoiseModel.from_backend(noisy_backend)
    noisy_simulator = AerSimulator(noise_model=noise_model)

    for N, shots in zip(N_list, shots_list):
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


                counts_old = test_QVAR_counts_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], shots=shots, backend=AerSimulator())
                counts_new = test_QVAR_counts_new(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], shots=shots, backend=AerSimulator())
                counts_old_noisy = test_QVAR_counts_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], shots=shots, backend=noisy_simulator)
                counts_new_noisy = test_QVAR_counts_new(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], shots=shots, backend=noisy_simulator)

                old_fidelity = hellinger_distance(counts_old, counts_old_noisy)
                new_fidelity = hellinger_distance(counts_new, counts_new_noisy)

                print("HDistance old: " + str(old_fidelity))
                print("HDistance new: " + str(new_fidelity))

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
    with open('results/results_uniform.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N', 'mean_hdistance_old', 'std_hdistance_old', 'mean_hdistance_new', 'std_hdistance_new'])  # Write header
        writer.writerows(rows)  # Write rows


def test_gaussian():
    N_list = [2,4,8]
    trial = 10
    seeds = 1
    shots_list = [100 * x for x in N_list]

    mean_trials_old = []
    mean_trials_new = []
    std_trials_old = []
    std_trials_new = []

    service = QiskitRuntimeService()
    noisy_backend = service.backend("ibm_torino")
    noise_model = NoiseModel.from_backend(noisy_backend)
    noisy_simulator = AerSimulator(noise_model=noise_model)

    for N, shots in zip(N_list, shots_list):
        print("N = " + str(N))
        cl = []
        q_old = []
        q_new = []
        
        n = math.ceil(math.log2(N))
        
        for t in range(trial):
            np.random.seed(123*t)
            vector = np.random.normal(0, 0.5, N)
            vector = np.clip(vector, -1, 1)
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

                counts_old = test_QVAR_counts_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], shots=shots, backend=AerSimulator())
                counts_new = test_QVAR_counts_new(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], shots=shots, backend=AerSimulator())
                counts_old_noisy = test_QVAR_counts_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], shots=shots, backend=noisy_simulator)
                counts_new_noisy = test_QVAR_counts_new(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], shots=shots, backend=noisy_simulator)

                old_fidelity = hellinger_distance(counts_old, counts_old_noisy)
                new_fidelity = hellinger_distance(counts_new, counts_new_noisy)

                print("HDistance old: " + str(old_fidelity))
                print("HDistance new: " + str(new_fidelity))

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
    with open('results/results_gaussian.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N', 'mean_hdistance_old', 'std_hdistance_old', 'mean_hdistance_new', 'std_hdistance_new'])  # Write header
        writer.writerows(rows)  # Write rows


def print_result(filename="results/results_uniform.csv"):
    df = pd.read_csv(filename)
    print(df)
    N_list = df['N']
    mean_hdistance_old = df['mean_hdistance_old']
    mean_hdistance_new = df['mean_hdistance_new']
    std_hdistance_old = df['std_hdistance_old']
    std_hdistance_new = df['std_hdistance_new']
    fig, ax = plt.subplots(figsize=(10,8))

    #color1 = '#f1a340'
    #color2 = '#998ec3'
    color1 = '#fc8d59'
    color2 = '#91bfdb'
    a = 0.3

    plt.rcParams.update({'font.size': 22})
    plt.tick_params(labelsize=22)

    mean_hdistance_old = np.array(mean_hdistance_old)
    mean_hdistance_new = np.array(mean_hdistance_new)
    std_hdistance_old = np.array(std_hdistance_old)
    std_hdistance_new = np.array(std_hdistance_new)

    x = [str(n) for n in N_list]

    ax.errorbar(x, mean_hdistance_old, yerr=std_hdistance_old, label='QVAR old', color=color1, ecolor=color1, capsize=10, capthick=2)
    line_1, = plt.plot(x, mean_hdistance_old, color=color1, label='QVAR old')
    fill_1 = plt.fill_between(x, mean_hdistance_old - std_hdistance_old, mean_hdistance_old + std_hdistance_old, color=color1, alpha=a)

    ax.errorbar(x, mean_hdistance_new, yerr=std_hdistance_new, label='QVAR new', color=color2, ecolor=color2, capsize=10, capthick=2)
    line_2, = plt.plot(x, mean_hdistance_new, color=color2, label='QVAR new')
    fill_2 = plt.fill_between(x, mean_hdistance_new - std_hdistance_new, mean_hdistance_new + std_hdistance_new, color=color2, alpha=a)

    plt.xlabel('N', fontsize=22)
    plt.ylabel('Hellinger Distance', fontsize=22)

    plt.legend([(line_1, fill_1), (line_2, fill_2)], ['QVAR old', 'QVAR new'])
    plt.savefig("plots/hellinger_distance.png")

if __name__ == "__main__":
    test_gaussian()
    print_result("results/results_gaussian.csv")
