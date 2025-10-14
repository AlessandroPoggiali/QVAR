from qiskit import *
from qiskit.circuit.library import *
from qiskit.circuit import *
from qiskit.quantum_info import *
from qiskit.circuit.library import UnitaryGate

import numpy as np
import math

from qiskit.quantum_info import *
import matplotlib.pyplot as plt

from qiskit_algorithms import EstimationProblem
from qiskit_algorithms import AmplitudeEstimation, FasterAmplitudeEstimation
from qiskit.primitives import Sampler

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

        counts = backend.run(transpile(qc, backend), shots=shots).result().get_counts()

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
            
        counts = backend.run(transpile(qc, backend), shots=shots).result().get_counts()

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
