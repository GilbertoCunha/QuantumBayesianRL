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
    
    def state_preparation(self, qr):
        # Create quantum circuit
        n_qubits = len(qr)
        circuit = QuantumCircuit(qr)
        
        # Apply controlled rotation gates to every qubit
        for i in range(n_qubits):
            # Get all rotation angles and control qubits
            angles = self.ry_angles[i]
            q_controls = [qr[j] for j in self.qbit_parents[i]]
            
            # Iterate all rotations
            for theta, states in angles:

                # Apply a multiple controlled rotation gate
                if len(q_controls) > 0:
                    for j in states:
                        if states[j] == 0:
                            circuit.x(j)
                    circuit.mcry(theta=theta, q_controls=q_controls, q_target=qr[i])
                    for j in states:
                        if states[j] == 0:
                            circuit.x(j)
                
                # Apply a rotation gate
                else:
                    circuit.ry(theta, qr[i])
            
        return circuit

    @staticmethod
    def grover_oracle(good_states, qr):
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
            qubits = list(range(len(good_states[0])))
            circuit.mcp(np.pi, qubits[:-1], qubits[-1])

            # Reflip unset qubits in good state
            for i, value in enumerate(state):
                if not int(value):
                    circuit.x(i)

        return circuit

    def grover_diffuser(qr):
        # Create quantum circuit
        circuit = QuantumCircuit(qr)

        # Apply hermitian conjugate state preparation circuit
        circuit.compose(self.state_preparation(qr).inverse(), inplace=True)

        # Flip about the zero state
        circuit.x(qr)
        circuit.mcp(np.pi, qr[:-1], qr[-1])
        circuit.x(qr)

        # Apply state preparation circuit
        circuit.compose(state_preparation, inplace=True)
        return circuit

    def grover_operator(good_states, qr):
        circuit = QuantumCircuit(qr)
        circuit.compose(self.grover_oracle(good_states, qr), inplace=True)
        circuit.compose(self.grover_diffuser(qr), inplace=True)
        return circuit

    def sample(self, query, evidence, n_samples=1000):

        # Number of grover operator iterations
        grover_iter = 1

        # Boolean flag to stop when evidence matches
        done = False

        # Define the good states

        # Define quantum register
        qr = QuantumRegister(len(self.name_to_qbit))

        # Define two classical registers
        # One for evidence qbits, one for the rest
        er = ClassicalRegister(len(evidence))
        cr = ClassicalRegister(n_qubits - len(evidence))

        # Apply grover's algorithm repeatedly until 
        # measured evidence qubit values match the given evidence
        while not done:

            # Create quantum circuit
            circuit = QuantumCircuit(qr, cr, er)

            # Apply state preparation
            circuit.compose(self.state_preparation(qr), inplace=True)

            # Apply grover operator multiple times
            for _ in range(grover_iter):
                circuit.compose(self.grover_operator(good_states, qr), inplace=True)

            # TODO: measure, check if evidence is correct, repeat