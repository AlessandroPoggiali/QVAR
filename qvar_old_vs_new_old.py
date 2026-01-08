from qiskit import *
from qiskit.circuit.library import *
from qiskit.circuit import *
from qiskit.quantum_info import *
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag
import math
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import *
from qiskit.quantum_info.analysis import hellinger_fidelity, hellinger_distance
from qiskit_ibm_runtime import QiskitRuntimeService
import matplotlib.pyplot as plt
import csv
import pandas as pd
from qiskit.transpiler.passes import Depth

from qiskit.quantum_info import *
import matplotlib.pyplot as plt

from qiskit_algorithms import EstimationProblem
from qiskit_algorithms import AmplitudeEstimation, FasterAmplitudeEstimation
from qiskit_ibm_runtime import Sampler

from qiskit.transpiler import CouplingMap, Layout
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix

# QVAR subroutine for computing the variance of a set of values encoded into a quantum state
#
#
# U: state preparation
# var_index: list of qubit indices of which we want to compute the variance. If more than
#            var_index qubits are present in U, the first var_index are the target qubits.
# ps_index: list of U's qubits that require a post selection measurement  
# version: method for estimating the variance
#
#    'FAE'    (default) Faster Amplitude Estimation
#    'AE'     Amplitude Estimation
#    'SHOTS'  measurements with multiple circuit execution
#
# delta (optional)          : target accuracy (FAE)
# max_iter (optional)       : maximum number of iterations of the oracle (FAE)
# eval_qbits (optional)     : number of additional qubits (AE)
# shots (optional)          : number of shots (SHOTS)
# n_h_gates (optional)      : normalization constant to multiply the final value
# postprocessing (optional) : if True, return the MLE postprocessed value (only for AE)
# backend (optional):       : specified backend on which the simulation will be run

def QVAR_old(U, var_index=None, ps_index=None, version='FAE', delta=0.0001, max_iter=5, eval_qbits=5, shots=8192, n_h_gates=0, postprocessing=True, backend=None):

    if var_index is None:
        var_index = [x for x in range(U.num_qubits)]
    
    i_qbits = len(var_index)
    q_qbits = e_qbits = i_qbits
    u_qbits = U.num_qubits

    a = QuantumRegister(1,'a')
    e = QuantumRegister(e_qbits,'e')
    q = QuantumRegister(q_qbits,'q')
    u = QuantumRegister(u_qbits, 'u')

    if version == 'SHOTS':
        ca = ClassicalRegister(1,'ca')
        cq = ClassicalRegister(q_qbits,'cq')
        ce = ClassicalRegister(e_qbits,'ce')
        if ps_index is not None:
            cps = ClassicalRegister(len(ps_index), 'cps')
            qc = QuantumCircuit(a, e, q, u, ca, cq, ce, cps)
        else:
            qc = QuantumCircuit(a, e, q, u, ca, cq, ce)
    
    else:
        qc = QuantumCircuit(a, e, q, u)

    #qc.append(U.to_gate(), list(range(1+e_qbits+q_qbits, qc.num_qubits)))    
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
    
    if ps_index is None:
        objective_qubits = [x for x in range(1+e_qbits+q_qbits)]
    else:
        objective_qubits = [x for x in range(1+e_qbits+q_qbits)]+[qc.num_qubits-u_qbits + x for x in ps_index]
    
    if version == 'SHOTS':
        qc.measure(a, ca) 
        qc.measure(q, cq)
        qc.measure(e, ce)
        
        if ps_index is not None:
            qc.measure(u[ps_index], cps)
            target_conf = '1'*len(ps_index) + ' ' + '1'*e_qbits + ' ' + '1'*q_qbits + ' 1' 
        else:
            target_conf = '1'*e_qbits + ' ' + '1'*q_qbits + ' 1'


        qc_t = transpile(qc, backend)    
        counts = backend.run(qc_t, shots=shots).result().get_counts()
        
        print("depth old: " + str(qc_t.depth()))
        print("size_old: " + str(qc_t.size()))
        #print("gate_fidelity_variance_old: " + str(backend._noise_info.gate_fidelity_variance))
        return counts
        try: 
            var = (counts[target_conf])/shots
        except:
            var = 0
            
    elif version == 'AE':
        sampler = Sampler()
        sampler.set_options(backend=backend)
        ae = AmplitudeEstimation(
            num_eval_qubits=eval_qbits,  
            sampler=sampler,
        )
        
        problem = EstimationProblem(
            state_preparation=qc, 
            objective_qubits=objective_qubits,
        )
        ae_result = ae.estimate(problem)

        if postprocessing:
            var = ae_result.mle
        else:
            var = ae_result.estimation
        
    elif version == 'FAE':
        sampler = Sampler()
        sampler.set_options(backend=backend)
        fae = FasterAmplitudeEstimation(
            delta=delta, 
            maxiter=max_iter,  
            sampler=sampler
        )

        problem = EstimationProblem(
            state_preparation=qc, 
            objective_qubits=objective_qubits,
        )
        fae_result = fae.estimate(problem)
        var = fae_result.estimation
    
    elif version == 'STATEVECTOR':
       
        problem = EstimationProblem(
            state_preparation=qc, 
            objective_qubits=objective_qubits,
        )

        if backend._noise_info:
            #density_matrix = DensityMatrix(qc)
            
            transpiled_circuit = transpile(qc, backend)
            transpiled_circuit.save_density_matrix()
            density_matrix = np.asarray(backend.run(transpiled_circuit).result().results[0].data.density_matrix)
            return density_matrix
        else:
            #statevector = Statevector(qc)
            
            transpiled_circuit = transpile(qc, backend)
            transpiled_circuit.save_statevector()
            statevector = np.asarray(backend.run(transpiled_circuit).result().get_statevector())
            
            return statevector
            '''
            var = 0
            for i, amplitude in enumerate(statevector):
                full_state = bin(i)[2:].zfill(qc.num_qubits)[::-1]
                state = ''.join([full_state[i] for i in objective_qubits])
                if problem.is_good_state(state[::-1]):
                    var = var + np.abs(amplitude) ** 2
            '''
    
    tot_hadamard = 2 + i_qbits + n_h_gates
    norm_factor = 2**tot_hadamard/2**i_qbits

    return var*norm_factor

def QVAR(U, var_index=None, ps_index=None, version='FAE', delta=0.0001, max_iter=5, eval_qbits=5, shots=8192, n_h_gates=0, postprocessing=True, backend=None):

    if var_index is None:
        var_index = [x for x in range(U.num_qubits)]
    
    i_qbits = len(var_index)
    e_qbits = i_qbits
    u_qbits = U.num_qubits

    a = QuantumRegister(1,'a')
    e = QuantumRegister(e_qbits,'e')
    u = QuantumRegister(u_qbits, 'u')

    if version == 'SHOTS':
        ca = ClassicalRegister(1,'ca')
        ce = ClassicalRegister(e_qbits,'ce')
        if ps_index is not None:
            cps = ClassicalRegister(len(ps_index), 'cps')
            qc = QuantumCircuit(a, e, u, ca, ce, cps)
        else:
            qc = QuantumCircuit(a, e, u, ca, ce)
    
    else:
        qc = QuantumCircuit(a, e, u)
    
    #qc.append(U.to_gate(), list(range(1+e_qbits, qc.num_qubits)))       
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

    if ps_index is None:
        objective_qubits = [x for x in range(1+e_qbits)]
    else:
        objective_qubits = [x for x in range(1+e_qbits)]+[qc.num_qubits-u_qbits + x for x in ps_index]

    if version == 'SHOTS':
        qc.measure(a, ca) 
        qc.measure(e, ce)
        
        if ps_index is not None:
            qc.measure(u[ps_index], cps)
            target_conf = '1'*len(ps_index) + ' ' + '1'*e_qbits + ' 1' 
        else:
            target_conf = '1'*e_qbits + ' 1'
        qc_t = transpile(qc, backend)    
        counts = backend.run(qc_t, shots=shots).result().get_counts()

        print("depth_new: " + str(qc_t.depth()))
        print("size_new: " + str(qc_t.size()))
        #print("gate_fidelity_variance_new: " + str(backend._noise_info.gate_fidelity_variance))
        return counts
        try: 
            var = (counts[target_conf])/shots
        except:
            var = 0
            
    elif version == 'AE':
        sampler = Sampler()
        sampler.set_options(backend=backend)
        ae = AmplitudeEstimation(
            num_eval_qubits=eval_qbits,  
            sampler=sampler
        )
        
        problem = EstimationProblem(
            state_preparation=qc, 
            objective_qubits=objective_qubits,
        )
        ae_result = ae.estimate(problem)
    
        if postprocessing:
            var = ae_result.mle
        else:
            var = ae_result.estimation
        
    elif version == 'FAE':
        sampler = Sampler()
        sampler.set_options(backend=backend)
        fae = FasterAmplitudeEstimation(
            delta=delta, 
            maxiter=max_iter,  
            sampler=sampler
        )

        problem = EstimationProblem(
            state_preparation=qc, 
            objective_qubits=objective_qubits,
        )
        fae_result = fae.estimate(problem)
        var = fae_result.estimation

    elif version == 'STATEVECTOR':
        problem = EstimationProblem(
            state_preparation=qc, 
            objective_qubits=objective_qubits,
        )

        if backend._noise_info:
            #density_matrix = DensityMatrix(qc)
            
            transpiled_circuit = transpile(qc, backend)
            transpiled_circuit.save_density_matrix()
            density_matrix = np.asarray(backend.run(transpiled_circuit).result().results[0].data.density_matrix)
            return density_matrix
        else:
            #statevector = Statevector(qc)
            
            transpiled_circuit = transpile(qc, backend)
            transpiled_circuit.save_statevector()
            statevector = np.asarray(backend.run(transpiled_circuit).result().get_statevector())
            
            return statevector
        
    
    tot_hadamard = 2 + n_h_gates
    norm_factor = 2**tot_hadamard/2**i_qbits

    return var*norm_factor

def _register_switcher(circuit, value, qubit_index):
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1]
    for idx, bit in enumerate(value):
        if not int(bit):
            circuit.x(qubit_index[idx])


def test():
    N_list = [64]
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

                '''
                sv_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(3*n+2, noise_info=False, seed=s*123))
                sv_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(2*n+2, noise_info=False, seed=s*123))
                dm_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(3*n+2, noise_info=True, seed=s*123))
                dm_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(2*n+2, noise_info=True, seed=s*123))
                '''

                counts_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', shots=shots, n_h_gates=n, backend=noisy_simulator)
                counts_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', shots=shots, n_h_gates=n, backend=noisy_simulator)
                counts_old_noisy = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', shots=shots, n_h_gates=n, backend=noisy_simulator)
                counts_new_noisy = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', shots=shots, n_h_gates=n, backend=noisy_simulator)

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


def test_gaussian():
    N_list = [64]
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
            vector = np.random.normal(0, 0.3, N)
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

                '''
                sv_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(3*n+2, noise_info=False, seed=s*123))
                sv_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(2*n+2, noise_info=False, seed=s*123))
                dm_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(3*n+2, noise_info=True, seed=s*123))
                dm_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(2*n+2, noise_info=True, seed=s*123))
                '''

                counts_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', shots=shots, n_h_gates=n, backend=AerSimulator())
                counts_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', shots=shots, n_h_gates=n, backend=AerSimulator())
                counts_old_noisy = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', shots=shots, n_h_gates=n, backend=noisy_simulator)
                counts_new_noisy = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', shots=shots, n_h_gates=n, backend=noisy_simulator)

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
        
    '''
    plt.scatter(N_list, mean_trials_old, label="Fidelity old")
    plt.scatter(N_list, mean_trials_new, label="Fidelity new")
    plt.xticks(N_list)
    plt.legend()
    plt.savefig("fidelity.png")
    '''

    rows = zip(N_list, mean_trials_old, std_trials_old, mean_trials_new, std_trials_new)

    # Write to a CSV file
    with open('results_g.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N', 'mean_fidelity_old', 'std_fidelity_old', 'mean_fidelity_new', 'std_fidelity_new'])  # Write header
        writer.writerows(rows)  # Write rows

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


def print_result(filename="results.csv"):
    df = pd.read_csv(filename)
    print(df)
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
    plt.ylabel('Hellinger Distance', fontsize=22)

    plt.legend([(line_1, fill_1), (line_2, fill_2)], ['QVAR old', 'QVAR new'])
    plt.savefig("fidelity.png")

if __name__ == "__main__":
    test()
    #print_result()
    #test_gaussian()
    #print_result("results_g.csv")