import numpy as np
import math
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister
from qvar import QVAR


def get_statevector(circuit):
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend=backend, shots=1024, memory=True)
    job_result = job.result()
    statevector = job_result.get_statevector(circuit)
    tol = 1e-16
    statevector = np.asarray(statevector)
    statevector.real[abs(statevector.real) < tol] = 0.0
    statevector.imag[abs(statevector.imag) < tol] = 0.0
    return statevector.real

def _register_switcher(circuit, value, qubit_index):
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1]
    for idx, bit in enumerate(value):
        if not int(bit):
            circuit.x(qubit_index[idx])

def test_random_U(u_size):

    c_variances = []
    q_variances = []

    print("Classical - Quantum")
    for _ in range(5):
        U = QuantumCircuit(u_size)
        U.h([x for x in range(u_size)])
        
        for z in range(u_size):
            U.ry(np.random.uniform(0, 2*np.pi), z)
        
        c_var = np.var(get_statevector(U))
        q_var = QVAR(U, version='FAE')
        
        print(str(c_var)+ ' - ' + str(q_var))
        
        c_variances.append(c_var)
        q_variances.append(q_var)


    differences = [(q-c)**2 for q,c in zip(q_variances, c_variances)]

    print("MSE: " + str(np.mean(differences)))

def test_ffqram(N):
    cl = []
    qu = []

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
            
        q_var = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='FAE', n_h_gates=n)
        classical = np.var(vector)
        print(str(classical)+ " - " + str(q_var))
        qu.append(q_var)
        cl.append(classical)

    differences = [(q-c)**2 for q,c in zip(qu, cl)]
    print("MSE: " + str(np.mean(differences)))


if __name__ == "__main__":

    
    

    print("\n RANDOM UNITARY TEST \n")
    test_random_U(2)

    print("\n FF-QRAM TEST \n")
    test_ffqram(8)