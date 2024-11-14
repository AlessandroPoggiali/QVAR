import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.fake_provider import *
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix
from qiskit.transpiler import CouplingMap
from qvar import QVAR, QVAR_old
import matplotlib.pyplot as plt
import csv


def _register_switcher(circuit, value, qubit_index):
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1]
    for idx, bit in enumerate(value):
        if not int(bit):
            circuit.x(qubit_index[idx])

def test_random_U(u_size):

    c_variances = []
    q_variances = []

    backend = AerSimulator() # noiseless backend

    print("Classical - Quantum")
    for _ in range(5):
        U = QuantumCircuit(u_size)
        U.h([x for x in range(u_size)])
        
        for z in range(u_size):
            U.ry(np.random.uniform(0, 2*np.pi), z)
        
        c_var = QVAR(U, version='STATEVECTOR', backend=backend)
        q_var = QVAR(U, version='FAE', backend=backend)
        
        print(str(c_var)+ ' - ' + str(q_var))
        
        c_variances.append(c_var)
        q_variances.append(q_var)


    differences = [(q-c)**2 for q,c in zip(q_variances, c_variances)]

    print("MSE: " + str(np.mean(differences)))

def test_ffqram(N):
    cl = []
    qu = []

    backend = AerSimulator() # noiseless backend

    print("Classical - Quantum")
    for _ in range(5):
        vector = np.random.uniform(-1,1, N)
        n = math.ceil(math.log2(N))

        r = QuantumRegister(1, 'r') 
        i = QuantumRegister(n, 'i')  

        U = QuantumCircuit(i, r)

        U.h(i)

        for index, val in zip(range(len(vector)), vector):
            _register_switcher(U, index, i)
            U.mcry(np.arcsin(val)*2, i[0:], r) 
            _register_switcher(U, index, i)
            
        q_var = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='FAE', n_h_gates=n, backend=backend)
        classical = np.var(vector)
        print(str(classical)+ " - " + str(q_var))
        qu.append(q_var)
        cl.append(classical)

    differences = [(q-c)**2 for q,c in zip(qu, cl)]
    print("MSE: " + str(np.mean(differences)))

def test_noise():

    N_list = [2,4,8,16,32]
    trial = 5
    seeds = 5

    mae_qold = []
    mae_qnew = []
    mse_qold = []
    mse_qnew = []

    for N in N_list:
        print("N = " + str(N))
        cl = []
        q_old = []
        q_new = []
        
        n = math.ceil(math.log2(N))
        
        for i in range(trial):
            np.random.seed(123*i)
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

                q_var_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', n_h_gates=n, backend=GenericBackendV2(3*n+2, noise_info=True, seed=s*123))
                q_var_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', n_h_gates=n, backend=GenericBackendV2(2*n+2, noise_info=True, seed=s*123))
                
                q_old_seed.append(q_var_old)
                q_new_seed.append(q_var_new)
                
            q_old.append(np.mean(q_old_seed))
            q_new.append(np.mean(q_new_seed))

        mae_old = np.mean([abs(x-y) for x,y in zip(q_old, cl)])
        mae_new = np.mean([abs(x-y) for x,y in zip(q_new, cl)])
        print("MAE QVAR_old = " + str(mae_old))
        print("MAE QVAR_new = " + str(mae_new))
        mae_qold.append(mae_old)
        mae_qnew.append(mae_new)

        mse_old = np.mean([(x-y)**2 for x,y in zip(q_old, cl)])
        mse_new = np.mean([(x-y)**2 for x,y in zip(q_new, cl)])
        print("MSE QVAR_old = " + str(mse_old))
        print("MSE QVAR_new = " + str(mse_new))
        mse_qold.append(mse_old)
        mse_qnew.append(mse_new)

    plt.scatter(N_list, mae_qold, label="MAE_QVAR_old")
    plt.scatter(N_list, mae_qnew, label="MAE_QVAR_new")
    plt.xticks(N_list)
    plt.legend()
    plt.savefig("MAE.png")
    fig2, ax2 = plt.subplots()
    plt.scatter(N_list, mse_qold, label="MSE_QVAR_old")
    plt.scatter(N_list, mse_qnew, label="MSE_QVAR_new")
    plt.xticks(N_list)
    plt.legend()
    plt.savefig("MSE.png")

def test_noise_density_matrix():

    N_list = [2,4,8,16,32]
    trial = 10
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
        
        for i in range(trial):
            np.random.seed(123*i)
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

                sv_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(3*n+2, noise_info=False, seed=s*123))
                sv_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(2*n+2, noise_info=False, seed=s*123))
                dm_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(3*n+2, noise_info=True, seed=s*123))
                dm_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=GenericBackendV2(2*n+2, noise_info=True, seed=s*123))
                

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
        writer.writerow(['N', 'mean_fidelity_old', 'std_trials_old', 'mean_trials_new', 'std_trials_new'])  # Write header
        writer.writerows(rows)  # Write rows
    
    print_result(N_list, mean_trials_old, mean_trials_new, std_trials_old, std_trials_new)

def print_result(N_list, mean_trials_old, mean_trials_new, std_trials_old, std_trials_new):
    fig, ax = plt.subplots(figsize=(10,8))

    #color1 = '#f1a340'
    #color2 = '#998ec3'
    color1 = '#fc8d59'
    color2 = '#91bfdb'
    a = 0.3

    plt.rcParams.update({'font.size': 22})
    plt.tick_params(labelsize=22)

    mean_trials_old = np.array(mean_trials_old)
    mean_trials_new = np.array(mean_trials_new)
    std_trials_old = np.array(std_trials_old)
    std_trials_new = np.array(std_trials_new)

    x = [str(n) for n in N_list]

    ax.errorbar(x, mean_trials_old, yerr=std_trials_old, label='QVAR old', color=color1, ecolor=color1, capsize=10, capthick=2)
    line_1, = plt.plot(x, mean_trials_old, color=color1, label='QVAR old')
    fill_1 = plt.fill_between(x, mean_trials_old - std_trials_old, mean_trials_old + std_trials_old, color=color1, alpha=a)

    ax.errorbar(x, mean_trials_new, yerr=std_trials_new, label='QVAR new', color=color2, ecolor=color2, capsize=10, capthick=2)
    line_2, = plt.plot(x, mean_trials_new, color=color2, label='QVAR new')
    fill_2 = plt.fill_between(x, mean_trials_new - std_trials_new, mean_trials_new + std_trials_new, color=color2, alpha=a)

    plt.xlabel('N', fontsize=22)
    plt.ylabel('Fidelity', fontsize=22)

    plt.legend([(line_1, fill_1), (line_2, fill_2)], ['QVAR old', 'QVAR new'])
    plt.savefig("fidelity.png")

if __name__ == "__main__":

    '''
    print("\n RANDOM UNITARY TEST \n")
    test_random_U(2)

    print("\n FF-QRAM TEST \n")
    test_ffqram(8)
    '''
    #test_noise()
    test_noise_density_matrix()