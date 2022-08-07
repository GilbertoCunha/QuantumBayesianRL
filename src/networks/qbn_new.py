from __future__ import annotations
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from src.utils import df_binary_str_filter, product_dict, counts_to_dict
from src.networks.bn import BayesianNetwork as BN
from qiskit.providers.aer import QasmSimulator
from math import log, ceil
from typing import Union
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
        self.grover_circ = self.grover_circ()
        
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
        
        if len(qubits[1::]) > 0:
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
    
    def grover_circ(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr)
        return circ
    
    def query(self, query: list[Id], evidence: dict[Id, Value], n_samples: int) -> pd.DataFrame:
        # Get the list of qubits to query
        query_qubits = sorted([j for nid in query for j in self.rv_qubits[nid]])
        measure_qubits = [self.qr[q] for q in query_qubits]
        
        # Create classical register for the measurement and quantum circuit
        cr = ClassicalRegister(len(query_qubits))
        circ = QuantumCircuit(self.qr, cr)
        
        # Apply encoding
        circ.compose(self.encoding_circ, inplace=True)
        
        # Perform measurement
        circ.measure(measure_qubits, cr)
        
        # Create simulator, job and get results
        simulator = QasmSimulator()
        job = simulator.run(circ, shots=n_samples)
        results = job.result().get_counts(circ)
        
        # Create dict of query rvs to list of measurement qubits
        query_rv_qubits = {rv: [] for rv in query}
        for i, q in enumerate(query_qubits):
            rv = self.qubit_to_id(q)
            query_rv_qubits[rv].append(i)
        
        return counts_to_dict(query, results, query_rv_qubits)