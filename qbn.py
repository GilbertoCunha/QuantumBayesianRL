from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
import numpy as np

class QuantumBayesianNetwork:
    def __init__(self, bn):
        # Dictionary mapping node names to qbit numbers
        self.name_to_qbit = {name: i for i, name in enumerate(bn.nodes)}
        
        # Dictionary mapping qbit numbers to node names
        self.qbit_to_name = {i: name for i, name in enumerate(bn.nodes)}
        
        # Dictionary mapping qbit numbers to a list of qbit parents (numbers also)
        self.qbit_parents = {self.name_to_qbit[name]: [self.name_to_qbit[p] for p in bn.graph[name].parents] for name in bn.nodes}
        
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
                ps = [(p1, p0, {self.name_to_qbit[n]: d[n] for n in d}) for (p1, p0, d) in ps]
            else:
                ps = [(pt[pt[name]==1]["Prob"].iloc[0], pt[pt[name]==0]["Prob"].iloc[0], {})]
                
            # Calculate list of rotations
            r[self.name_to_qbit[name]] = [(angle(p1, p0), states) for p1, p0, states in ps]
                
        return r
    
    def build(self):
        # Create quantum circuit
        n_qubits = len(self.name_to_qbit)
        qr, cr = QuantumRegister(n_qubits), ClassicalRegister(n_qubits)
        circuit = QuantumCircuit(qr, cr)
        
        # Apply controlled rotation gates to every qubit
        for i in range(n_qubits):
            # Get all rotation angles and control qubits
            angles = self.ry_angles[i]
            q_controls = [qr[j] for j in self.qbit_parents[i]]
            
            # Iterate all rotations
            for theta, states in angles:
                if len(q_controls) > 0:
                    for j in states:
                        if states[j] == 0:
                            circuit.x(j)
                    circuit.mcry(theta=theta, q_controls=q_controls, q_target=qr[i])
                    for j in states:
                        if states[j] == 0:
                            circuit.x(j)
                else:
                    circuit.ry(theta, qr[i])
        
        circuit.measure(qr, cr)
            
        return circuit