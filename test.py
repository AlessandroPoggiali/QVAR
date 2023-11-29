import numpy as np
from qiskit import QuantumCircuit, execute, Aer
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

if __name__ == "__main__":

    u_size = 2

    c_variances = []
    q_variances = []

    for t in range(5):
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
