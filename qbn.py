from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import math


class QuantumBayesianNetwork:

    def __init__(self, bn):
        
        # Dictionary mapping qbit numbers to a list of qbit parents (numbers also)
        self.qbit_parents = {name: bn.graph[name].parents for name in bn.nodes}
        
        # Dictionary where keys are qubit numbers and values are lists of tuples
        # First element of each tuple an the angle of rotation
        # Second element is a dictionary that represents the state of the control qubits (1 if set, 0 if not)
        self.ry_angles = self.get_ry_angles(bn)
        
    def get_ry_angles(self, bn):
        # Create result dictionary
        r = {}
        
        # Define function to get the angle
        angle = lambda p1, p0: np.pi if (p0 == 0) else 2 * np.arctan(np.sqrt(p1 / p0))
        
        # Iterate nodes of Bayesian Network
        for name in bn.nodes:
            
            # Get node, parents and CPT
            node = bn.graph[name]
            parents = node.parents
            pt = node.pt
            
            # Find all p1 and p0 probabilities 
            # This is a list of (p1, p0) tuples
            if len(parents) > 0:
                ps = [(
                    group[group[name]==1]["Prob"].iloc[0], 
                    group[group[name]==0]["Prob"].iloc[0],
                    dict(group[parents].iloc[0])
                ) for _, group in pt.groupby(parents)]
            else:
                ps = [(pt[pt[name]==1]["Prob"].iloc[0], pt[pt[name]==0]["Prob"].iloc[0], {})]
                
            # Calculate list of rotations
            r[name] = [(angle(p1, p0), states) for p1, p0, states in ps]
                
        return r
    
    def state_preparation(self, qr, names_to_qbits):
        # Create quantum circuit
        n_qubits = len(qr)
        circuit = QuantumCircuit(qr)
        
        # Get the inverse dictionary
        qbits_to_names = {v: k for k, v in names_to_qbits.items()}
        
        # Apply controlled rotation gates to every qubit
        for i in range(n_qubits):
            i_name = qbits_to_names[i]
            
            # Get all rotation angles and control qubits
            angles = self.ry_angles[i_name]
            q_controls = [qr[names_to_qbits[j_name]] for j_name in self.qbit_parents[i_name]]
            
            # Iterate all rotations
            for theta, states in angles:

                # Apply a multiple controlled rotation gate
                if len(q_controls) > 0:
                    for name in states:
                        if states[name] == 0:
                            circuit.x(names_to_qbits[name])
                    circuit.mcry(theta=theta, q_controls=q_controls, q_target=qr[i])
                    for name in states:
                        if states[name] == 0:
                            circuit.x(names_to_qbits[name])
                
                # Apply a rotation gate
                else:
                    circuit.ry(theta, qr[i])
            
        return circuit

    @staticmethod
    def grover_oracle(qr, good_states):
        # good_states = ["0010", "1011", "1100"]

        # Create quantum circuit
        circuit = QuantumCircuit(qr)

        # Iterate each good state
        for state in good_states:

            # Flip unset qubits in good state
            for i, value in enumerate(state):
                if not int(value):
                    circuit.x(i)

            # Apply phase flip
            circuit.mcp(np.pi, qr[:-1], qr[-1])

            # Reflip unset qubits in good state
            for i, value in enumerate(state):
                if not int(value):
                    circuit.x(i)

        return circuit

    def grover_diffuser(self, qr, names_to_qbits):
        # Create quantum circuit
        circuit = QuantumCircuit(qr)

        # Apply hermitian conjugate state preparation circuit
        circuit.compose(self.state_preparation(qr, names_to_qbits).inverse(), inplace=True)
        circuit.barrier()

        # Flip about the zero state
        circuit.x(qr)
        circuit.mcp(np.pi, qr[:-1], qr[-1])
        circuit.x(qr)
        circuit.barrier()
        
        # Apply state preparation circuit
        circuit.compose(self.state_preparation(qr, names_to_qbits), inplace=True)
        return circuit

    def grover_operator(self, qr, good_states, names_to_qbits):
        circuit = QuantumCircuit(qr)
        circuit.compose(self.grover_oracle(qr, good_states), inplace=True)
        circuit.barrier()
        circuit.compose(self.grover_diffuser(qr, names_to_qbits), inplace=True)
        return circuit
    
    @staticmethod
    def bitGen(n):
        return [''.join(i) for i in itertools.product('01', repeat=n)]

    def query(self, query, evidence={}, n_samples=1000):
        
        # Number of shots of the circuit
        shots = n_samples if (len(evidence) == 0) else 1
        
        # Define qbit names
        evidence_names = [k for k in evidence]
        other_names = [name for name in self.qbit_parents if name not in evidence_names]
        names = other_names + evidence_names
        names_to_qbits = {name: i for i, name in enumerate(names)}
        qbits_to_names = {v: k for k, v in names_to_qbits.items()}

        # Define the good states
        other_states = self.bitGen(len(other_names))
        evidence_state = ''.join([str(evidence[k]) for k in evidence])
        good_states = [o + evidence_state for o in other_states]

        # Define quantum and classical registers
        qr = QuantumRegister(len(names_to_qbits))
        cr = ClassicalRegister(len(names_to_qbits))
        
        # Initialize samples
        samples = {name: [] for name in other_names}

        # Get the required number of samples
        iterations = n_samples if (len(evidence) != 0) else 1
        for _ in tqdm(range(iterations), total=iterations, desc="Sampling", leave=False):
        
            # Constants for number of grover iterations
            c = 1.4
            l = 1

            # Boolean flag to stop when evidence matches
            done = False
            
            # Apply grover's algorithm repeatedly until 
            # measured evidence qubit values match the given evidence
            while not done:
                # Update grover iterations
                m = int(math.ceil(c**l))
                grover_iter = int(np.random.randint(1, m+1))
                
                # Create quantum circuit
                circuit = QuantumCircuit(qr, cr)

                # Apply state preparation
                circuit.compose(self.state_preparation(qr, names_to_qbits), inplace=True)
                circuit.barrier()

                # Apply grover operator multiple times
                if len(evidence) != 0:
                    for _ in range(grover_iter):
                        circuit.compose(self.grover_operator(qr, good_states, names_to_qbits), inplace=True)
                    circuit.barrier()

                # Apply measurements
                circuit.measure(qr, cr)

                # Perform one measurement
                simulator = QasmSimulator()
                job = simulator.run(circuit, shots=shots)
                counts = job.result().get_counts(circuit)
                result = list(counts.keys())[0] if (len(evidence) != 0) else counts
                
                # Evidence and query measurements
                if len(evidence) != 0:
                    ev_meas = result[:len(evidence)][::-1]
                    que_meas = result[len(evidence):][::-1]

                # If evidence matches, append sample
                if (len(evidence) != 0) and (ev_meas == evidence_state):
                    # Get evidence and non evidence measurement
                    for i, v in enumerate(que_meas):
                        samples[qbits_to_names[i]].append(int(v))
                    done = True
                    
                # Otherwise double the number of grover operators applied
                elif len(evidence) != 0:
                    l += 1
                   
                # When there is no evidence
                else:
                    for key in result:
                        for i, v in enumerate(key[::-1]):
                            for _ in range(result[key]):
                                samples[qbits_to_names[i]].append(int(v))
                    done = True
                    
        # Turn samples into probability table
        df = pd.DataFrame(samples)
        df = df.value_counts(normalize=True).to_frame("Prob")
        
        # Group over query variables and sum over all other variables
        df = df.groupby(query).sum()
            
        return df