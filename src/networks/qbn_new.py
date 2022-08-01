from __future__ import annotations
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from src.networks.bn import BayesianNetwork as BN
from qiskit.providers.aer import QasmSimulator
from src.utils import df_binary_str_filter
from math import log, ceil
from typing import Union
import pandas as pd

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
        
    def encoding_circ(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr)
        return circ
    
    def grover_circ(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr)
        return circ
    
    def query(self, query: list[Id], evidence: dict[Id, Value], n_samples: int) -> pd.DataFrame:
        pass