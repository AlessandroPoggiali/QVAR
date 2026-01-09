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
from sklearn.preprocessing import MinMaxScaler
from qiskit.transpiler.passes import Depth
from qiskit.quantum_info import *
import matplotlib.pyplot as plt
from qiskit_algorithms import EstimationProblem
from qiskit_algorithms import AmplitudeEstimation, FasterAmplitudeEstimation
from qiskit_ibm_runtime import Sampler
from qiskit.transpiler import CouplingMap, Layout
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix
from sklearn.datasets import load_iris, load_wine, make_moons, load_breast_cancer
import os

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

def load_breast_cancer_dataset():
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    return X, y

def load_iris_dataset():
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target
    # Scale features to [0,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    return X, y

def load_wine_dataset():
    data = load_wine()
    X = data.data.astype(np.float32)
    y = data.target

    # Scale features to [0,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    return X, y 

# -----------------------------
# Aggregate metrics
# -----------------------------
def aggregate_metrics(abs_err, rel_err):
    return {
        'MAE': np.mean(abs_err),
        'RMSE': np.sqrt(np.mean(abs_err**2)),
        'Mean Relative Error': np.mean(rel_err)
    }

# -----------------------------
# Compute per-feature errors
# -----------------------------
def compute_errors(var_noisy, var_classical):
    abs_err = np.abs(var_noisy - var_classical)
    rel_err = abs_err / (var_classical + 1e-12)
    signed_err = var_noisy - var_classical
    return abs_err, rel_err, signed_err

def test_real():

    service = QiskitRuntimeService()
    noisy_backend = service.backend("ibm_torino")
    noise_model = NoiseModel.from_backend(noisy_backend)
    noisy_simulator = AerSimulator(noise_model=noise_model)

    n_samples = 32
    n = math.ceil(math.log2(n_samples))
    shots = n_samples * 100
    n_trials = 10
    #n_features = 5
    #X = np.random.uniform(-1,1, n_samples * n_features).reshape(n_samples, n_features) 

    X, y = load_wine_dataset()
    #X, y = load_breast_cancer_dataset()
    #X, y = load_iris_dataset()
    filename = "wine.csv"
    # CSV columns: trial, old MAE, old RMSE, old MRE, opt MAE, opt RMSE, opt MRE, impr_MAE, impr_RMSE, impr_MRE
    
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "trial",
                "MAE_old", "RMSE_old", "MRE_old",
                "MAE_opt", "RMSE_opt", "MRE_opt",
                "impr_MAE_pct", "impr_RMSE_pct", "impr_MRE_pct"
            ])

    for _trial in range(n_trials):
        print(f"\nTrial {_trial+1}/{n_trials}\n-----------------------------")
        np.random.seed(123*_trial)

        indices = np.random.choice(X.shape[0], size=n_samples, replace=False)  # without replacement
        X = X[indices]
        n_features = X.shape[1]

        var_classical = [] 
        var_old = []
        var_opt = []

        for feature in range(n_features):
            print(f"Processing feature {feature+1}/{n_features}...")
            feature_samples = X[:, feature].reshape(-1, 1)
            #scaler = MinMaxScaler()
            #feature_samples = scaler.fit_transform(feature_samples)
            feature_samples = feature_samples.flatten()
            feature_samples = feature_samples.clip(-0.9999, 0.9999)
            classical = np.var(feature_samples)
            var_classical.append(classical)

            r = QuantumRegister(1, 'r') 
            i = QuantumRegister(n, 'i')  
        
            U = QuantumCircuit(i, r)
            
            U.h(i)
            
            for index, val in zip(range(len(feature_samples)), feature_samples):
                _register_switcher(U, index, i)
                U.mcry(np.arcsin(val)*2, i[0:], r) 
                _register_switcher(U, index, i)

            old = QVAR_old(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', shots=shots, n_h_gates=n, backend=noisy_simulator)
            opt = QVAR(U, var_index=list(range(n)), ps_index=[U.num_qubits-1], version='SHOTS', shots=shots, n_h_gates=n, backend=noisy_simulator)
            print(f"Feature {feature}: Classical Variance = {classical:.4f}, QVAR_old = {old:.4f}, QVAR_opt = {opt:.4f}")
            var_old.append(old)
            var_opt.append(opt)

        var_classical = np.array(var_classical)
        var_old = np.array(var_old)
        var_opt = np.array(var_opt)

        abs_err_old, rel_err_old, _ = compute_errors(var_old, var_classical)
        abs_err_opt, rel_err_opt, _ = compute_errors(var_opt, var_classical)

        

        metrics_old = aggregate_metrics(abs_err_old, rel_err_old)
        metrics_opt = aggregate_metrics(abs_err_opt, rel_err_opt)

        # -----------------------------
        # Improvement metric per-feature
        # -----------------------------
        improvement_perc = 100 * (abs_err_old - abs_err_opt) / (abs_err_old + 1e-12)

        # Aggregate improvement
        impr_MAE = 100 * (metrics_old['MAE'] - metrics_opt['MAE']) / (metrics_old['MAE'] + 1e-12)
        impr_RMSE = 100 * (metrics_old['RMSE'] - metrics_opt['RMSE']) / (metrics_old['RMSE'] + 1e-12)
        impr_MRE = 100 * (metrics_old['Mean Relative Error'] - metrics_opt['Mean Relative Error']) / (metrics_old['Mean Relative Error'] + 1e-12)

        # CSV columns: trial, old MAE, old RMSE, old MRE, opt MAE, opt RMSE, opt MRE, impr_MAE, impr_RMSE, impr_MRE
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
            _trial + 1,
            metrics_old["MAE"], metrics_old["RMSE"], metrics_old["Mean Relative Error"],
            metrics_opt["MAE"], metrics_opt["RMSE"], metrics_opt["Mean Relative Error"],
            impr_MAE, impr_RMSE, impr_MRE
            ])

        '''
        print("Metrics qvar_old:", metrics_old)
        print("Metrics qvar_opt:", metrics_opt)
        print(f"Aggregate improvement MAE: {impr_MAE:.2f}%")
        print(f"Aggregate improvement RMSE: {impr_RMSE:.2f}%")
        print(f"Aggregate improvement Mean Relative Error: {impr_MRE:.2f}%")
        '''
    
    df = pd.read_csv(filename)
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)
    for col in df.columns[1:]:  # Skip 'trial' column
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"{col:20s} | Mean: {mean_val:8.4f} | Std: {std_val:8.4f}")
    
    with open("final_result.txt", "a") as result_file:
        result_file.write("Statistics for dataset: " + filename + "\n")
        result_file.write("="*50 + "\n")
        for col in df.columns[1:]:  # Skip 'trial' column
            mean_val = df[col].mean()
            std_val = df[col].std()
            result_file.write(f"{col:20s} | Mean: {mean_val:8.4f} | Std: {std_val:8.4f}\n")


if __name__ == "__main__":
    test_real()