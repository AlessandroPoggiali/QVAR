import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.fake_provider import *
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit.transpiler import CouplingMap
from qvar import QVAR, QVAR_old
import matplotlib.pyplot as plt


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

    N_list = [2,4,8,16,32,64]
    trial = 10

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
        backend = GenericBackendV2(3*n + 2)
        #print("Classical - QVAR_old - QVAR_new")
        for i in range(trial):
            np.random.seed(123*i)
            vector = np.random.uniform(-1,1, N)
            
        
            r = QuantumRegister(1, 'r') 
            i = QuantumRegister(n, 'i')  
        
            U = QuantumCircuit(i, r)
            
            U.h(i)
            
            for index, val in zip(range(len(vector)), vector):
                _register_switcher(U, index, i)
                U.mcry(np.arcsin(val)*2, i[0:], r) 
                _register_switcher(U, index, i)
                       
            q_var_old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=backend)
            q_var_new = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='STATEVECTOR', n_h_gates=n, backend=backend)
            classical = np.var(vector)
            #print(str(classical)+ " - " + str(q_var_old) + " - " + str(q_var_new))
            q_old.append(q_var_old)
            q_new.append(q_var_new)
            cl.append(classical)

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

if __name__ == "__main__":

    '''
    print("\n RANDOM UNITARY TEST \n")
    test_random_U(2)

    print("\n FF-QRAM TEST \n")
    test_ffqram(8)
    '''

    test_noise()