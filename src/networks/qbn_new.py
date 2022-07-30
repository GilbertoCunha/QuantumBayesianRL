from __future__ import annotations
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from src.networks.ddn import DynamicDecisionNetwork as DDN
from qiskit.providers.aer import QasmSimulator
from math import log, ceil
from typing import Union
import pandas as pd

# Define the types
Id = tuple[str, int]
Value = Union[int, float]


class QuantumDDN(DDN):
    
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
        
    def encoding_circ(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr)
        return circ
    
    def grover_circ(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr)
        return circ
    
    def query(self, query: list[Id], evidence: dict[Id, Value], n_samples: int) -> pd.DataFrame:
        pass