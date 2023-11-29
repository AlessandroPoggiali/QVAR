from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, BasicAer
from qiskit.algorithms import EstimationProblem
from qiskit.algorithms import AmplitudeEstimation, FasterAmplitudeEstimation
from qiskit.primitives import Sampler


# U: state preparation
# var_index: list of qubit indices of which we want to compute the variance
# version: method for estimating the variance
#
#    'FAE'    (default) Faster Amplitude Estimation
#    'AE'     Amplitude Estimation
#    'SHOTS'  measurements with multiple circuit execution
#
# delta (optional)                : target accuracy (FAE)
# max_iter (optional)             : maximum number of iterations of the oracle (FAE)
# eval_qbits (optional)           : number of additional qubits (AE)
# shots (optional)                : number of shots (SHOTS)
# normalization_factor (optional) : multiplicative constant to obtain the final value
# postprocessing (optional)       : if True, return the MLE postprocessed value (only valid for AE)
def QVAR(U, var_index=None, version='FAE', delta=0.0001, max_iter=5, eval_qbits=5, shots=8192, normalization_factor=None, postprocessing=False):

    if var_index == None:
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
        qc = QuantumCircuit(a, e, q, u, ca, cq, ce)
    
    else:
        qc = QuantumCircuit(a, e, q, u)

    qc.append(U.to_gate(), [qc.num_qubits - i - 1 for i in range(u_qbits)]) 
    
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
    
    if version == 'SHOTS':
        qc.measure(a, ca) 
        qc.measure(q, cq)
        qc.measure(e, ce)   

        backend = BasicAer.get_backend('qasm_simulator')
        counts = execute(qc, backend, shots=shots).result().get_counts(qc)
        target_conf = '1'*e_qbits + ' ' + '1'*q_qbits + " 1"

        try: 
            var = (counts[target_conf])/shots
        except:
            var = 0
            
    elif version == 'AE':
        sampler = Sampler()
        ae = AmplitudeEstimation(
            num_eval_qubits=eval_qbits,  # the number of evaluation qubits specifies circuit width and accuracy
            sampler=sampler
        )

        problem = EstimationProblem(
            state_preparation=qc,  # A operator
            objective_qubits=[x for x in range(qc.num_qubits-i_qbits)],
        )
        ae_result = ae.estimate(problem)

        if postprocessing:
            var = ae_result.mle
        else:
            var = ae_result.estimation
        
    elif version == 'FAE':
        sampler = Sampler()
        fae = FasterAmplitudeEstimation(
            delta=delta,  # target accuracy
            maxiter=max_iter,  # determines the maximal power of the Grover operator
            sampler=sampler,
        )

        problem = EstimationProblem(
            state_preparation=qc,  # A operator
            objective_qubits=[x for x in range(qc.num_qubits-i_qbits)],
        )
        fae_result = fae.estimate(problem)
        var = fae_result.estimation

    if normalization_factor is not None:
        var = var*normalization_factor
    else:    
        n_hadamard = 2+i_qbits
        norm_factor = 2**n_hadamard/2**i_qbits
        var = var*norm_factor

    return var