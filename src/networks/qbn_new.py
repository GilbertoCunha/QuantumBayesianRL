from __future__ import annotations
from src.utils import df_binary_str_filter, product_dict, get_string_elems
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from src.networks.bn import BayesianNetwork as BN
from qiskit.providers.aer import QasmSimulator
from math import log, ceil
from typing import Union
from tqdm import tqdm
import pandas as pd
import numpy as np

# Define the types
Id = tuple[str, int]
Value = Union[int, float]


class QuantumBayesianNetwork(BN):
    
    def initialize(self):
        # Initialize DDN parent class
        super().initialize()
        
        # Define Random Variable (DiscreteNode) to qubit dict
        self.rv_qubits = self.get_rv_qubit_dict()
        
        # Define quantum register
        n_qubits = sum([len(self.rv_qubits[key]) for key in self.rv_qubits])
        self.qr = QuantumRegister(n_qubits)
        
        # Create quantum circuits for running the queries
        self.encoding_circ = self.encoding_circ()
        self.grover_diffuser = self.grover_diffuser()
        
    def get_rv_qubit_dict(self) -> dict[Id, list[int]]:
        # Iterate nodes (already in topological order)
        counter, r = 0, {}
        for nid in self.node_queue:
            # Calculate number of qubits for the random variable
            value_space = self.node_dict[nid].get_value_space()
            n_qubits = ceil(log(len(value_space), 2))
            
            # Add list of qubits to the random variable qubit dict
            r[nid] = [counter + i for i in range(n_qubits)]
            counter += n_qubits
        return r
    
    def qubit_to_id(self, qubit: int) -> Id:
        return [key for key in self.rv_qubits if qubit in self.rv_qubits[key]][0]
    
    def count_to_dict(self, key: str, value: int, indices_dict: dict[Id, list[int]]) -> dict[str, int]:
        # TODO: Convert str_slice using the value space of each RV
        
        # Invert key (Qiskit is little endian)
        key = key[::-1]
        
        # Create result dictionary
        r = {}
        for rv, indices in indices_dict.items():
            # Re-invert string to grab value for each rv
            str_slice = get_string_elems(key, indices)[::-1]
            index = int(str_slice, 2)
            r[rv] = self.get_node(rv).get_value_space()[index]
            
        # Add probability entry (non-normalized)
        r["Prob"] = value
        
        return r
            
    def counts_to_dict(self, query: list[Id], results: dict[str, int]) -> pd.DataFrame:
        indices_dict = {rv: self.rv_qubits[rv] for rv in query}
        
        # Convert counts dict
        r = {k: [] for k in query}
        r["Prob"] = []
        for k, v in results.items():
            entry = self.count_to_dict(k, v, indices_dict)
            for k_, v_ in entry.items():
                r[k_].append(v_)
                
        # Create df of results
        df = pd.DataFrame(r).groupby(query).sum().reset_index()
        df["Prob"] /= df["Prob"].sum()
        df = df.sort_values(list(df.columns)).reset_index()
        
        # Remove index column if it exists
        if "index" in df:
            df = df.drop("index", axis=1)
            
        return df
    
    def qubits_prob(self, qubit_values: dict[int, int], rv_id: Id) -> float:
        """
        Calculates the probability that qubits have certain values.
        """
        
        # Dict of ids to qubit values dict
        id_values = {}
        for q, v in qubit_values.items():
            # Get id of qubit
            nid = self.qubit_to_id(q)
            
            # Select qubit position in binary representation
            index = q - min(self.rv_qubits[nid])
            
            # Add to dict
            if nid not in id_values:
                id_values[nid] = {index: v}
            else:
                id_values[nid][index] = v
        
        # Filter dataframe entries to the qubit values
        df = self.get_node(rv_id).get_pt()
        for nid in id_values:
            value_space = self.get_node(nid).get_value_space()
            df = df_binary_str_filter(df, nid, id_values[nid], value_space)
        
        return df["Prob"].sum()
    
    def recursive_rotation(self, qubits: list[int], parent_values: dict[int, int], nid: Id) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr)
        
        # Select current qubit
        q = qubits[0]
        
        # Calculate probabilities
        angle = lambda p1, p0: np.pi if (p0 == 0) else 2 * np.arctan(np.sqrt(p1 / p0))
        parents_0 = {**{q: 0}, **parent_values}
        p0 = self.qubits_prob(parents_0, nid)
        parents_1 = {**{q: 1}, **parent_values}
        p1 = self.qubits_prob(parents_1, nid)
        theta = angle(p1, p0)
        
        # Apply rotation gate
        if len(parent_values) == 0:
            circ.ry(theta, self.qr[q])
        else:
            q_controls = [self.qr[i] for i in parent_values.keys()]
            circ.mcry(theta, q_controls, self.qr[q])
        
        if len(qubits) > 1:
            # Recursive call to compose other rotations
            circ.compose(self.recursive_rotation(qubits[1::], parents_1, nid), inplace=True)
            
            # Apply not gate
            if len(parent_values) == 0:
                circ.x(self.qr[q])
            else:
                circ.mcx(q_controls, self.qr[q])
                
            # Recursive call to compose other rotations
            circ.compose(self.recursive_rotation(qubits[1::], parents_0, nid), inplace=True)
            
            # Apply not gate
            if len(parent_values) == 0:
                circ.x(self.qr[q])
            else:
                circ.mcx(q_controls, self.qr[q])
        
        return circ
        
    def encoding_circ(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr)
        
        # Iterate every random variable
        for nid in self.node_queue:
            # Get parent and RV qubits
            parents = [j for i in self.get_parents(nid) for j in self.rv_qubits[i]]
            qubits = self.rv_qubits[nid]
            
            # Iterate all possible values of parents
            parent_value_space = {q: [0, 1] for q in parents}
            iterator = product_dict(parent_value_space) if len(parents) > 0 else [{}]
            for parent_values in iterator:
                unset_parents = [p for p in parent_values if parent_values[p]==0]
                for p in unset_parents:
                    circ.x(self.qr[p])
                circ.compose(self.recursive_rotation(qubits, parent_values, nid), inplace=True)
                for p in unset_parents:
                    circ.x(self.qr[p])
        
        return circ
    
    def grover_oracle(self, evidence_qvalues: dict[int, int]) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr)
        
        # Flip unset qubits
        for q, value in evidence_qvalues.items():
            if value == 0:
                circ.x(self.qr[q])

        # Phase flip evidence qubits
        start = list(evidence_qvalues.keys())[0]
        if len(evidence_qvalues) == 1:
            circ.z(self.qr[start])
        else:
            controls = [self.qr[q] for q in evidence_qvalues]
            target = controls.pop()
            circ.mcp(np.pi, controls, target)

        # Flip unset qubits
        for q, value in evidence_qvalues.items():
            if value == 0:
                circ.x(self.qr[q])        
        
        return circ
    
    def grover_diffuser(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr)
        
        # Apply inverse encoding
        circ.compose(self.encoding_circ.inverse(), inplace=True)
        
        # Flip about the zero state
        circ.x(self.qr)
        circ.mcp(np.pi, self.qr[:-1], self.qr[-1])
        circ.x(self.qr)
        
        # Apply encoding
        circ.compose(self.encoding_circ, inplace=True)    
    
        return circ
    
    def grover_circ(self, evidence_qvalues: dict[int, int]) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr)
        circ.compose(self.grover_oracle(evidence_qvalues), inplace=True)
        circ.compose(self.grover_diffuser, inplace=True)
        return circ
    
    def query_circ(self, cr: ClassicalRegister, evidence_qvalues: dict[Id, int], grover_iter: int) -> QuantumCircuit:
        # Build circuit
        circ = QuantumCircuit(self.qr, cr)
        circ.compose(self.encoding_circ, inplace=True)
        if len(evidence_qvalues) != 0:
            for _ in range(grover_iter):
                circ.compose(self.grover_circ(evidence_qvalues), inplace=True)
                
        return circ
    
    def query(self, query: list[Id], evidence: dict[Id, Value], n_samples: int) -> pd.DataFrame:
        # Generate evidence qubit values dict
        evidence_qvalues = {}
        for rv, value in evidence.items():
            # Evidence bitstring (Maybe has to be inverted)
            bitstr = bin(self.get_node(rv).get_value_space().index(value))[2::]
            
            # Add qubits and values to evidence qvalues dict
            for q, v in zip(self.rv_qubits[rv], bitstr):
                evidence_qvalues[q] = v
        
        # Get the list of qubits for query and evidence
        query_qubits = sorted([j for nid in query for j in self.rv_qubits[nid]])
        evidence_qubits = sorted([j for nid in evidence for j in self.rv_qubits[nid]])
        
        # Define shots and number of iterations
        results = {}
        iterations = n_samples if len(evidence) > 0 else 1
        shots = 1 if len(evidence) > 0 else n_samples
        for _ in tqdm(range(iterations), total=iterations, desc="Sampling", leave=True):
            c = 1.4
            l = 1
            done = False
            
            while not done:
                # Update grover iterations
                m = int(ceil(c**l))
                grover_iter = int(np.random.randint(1, m+1))
                
                # Create classical register for the measurement and quantum circuit
                cr = ClassicalRegister(len(self.qr))
                circ = self.query_circ(cr, evidence_qvalues, grover_iter)
                circ.measure(self.qr, cr)
                
                # Create simulator, job and get results
                simulator = QasmSimulator()
                job = simulator.run(circ, shots=shots)
                counts = job.result().get_counts(circ)
                
                # If evidence needs to be checked
                if shots == 1:
                    done = True
                    bitstr = list(counts.keys())[0]
                    invbitstr = bitstr[::-1]
                    
                    # Check if evidence matches
                    evidence_measurements = {q: invbitstr[q] for q in evidence_qubits}
                    if evidence_measurements != evidence_qvalues:
                        done = False
                    elif bitstr not in results:
                        results[bitstr] = 1
                    else:
                        results[bitstr] += 1
                # No evidence
                else:
                    results = counts
                    done = True
        
        return self.counts_to_dict(query, results)