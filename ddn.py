from __future__ import annotations
from nodes import StateNode, EvidenceNode, ActionNode, UtilityNode
from typing import Type, Union
import networkx as nx
import pandas as pd
import numpy as np

# Defining types
Id = (str, int)
Edge = (Id, Id)
Node = Union[StateNode, EvidenceNode, ActionNode, UtilityNode]


class DynamicDecisionNetwork:
    def __init__(self):
        self.node_map: dict[Id, Node] = {}
        self.graph: dict[Id, [Id]] = {}
        self.knowns: list[Id, int] = {}
        self.time: int = 0
            
        
    def add_nodes(self, nodes: list[Node]):
        for node in nodes:
            if node.get_id() not in self.node_map:
                self.node_map[node.get_id()] = node
                self.graph[node.get_id()] = []
                
        
    def add_edges(self, edges: list[Edge]):
        for edge in edges:
            s, d = edge
            if s not in self.graph:
                self.graph[s] = []
            self.graph[s].append(d)
            
            
    def draw(self):
        # Create nx graph
        G = nx.DiGraph(directed=True)
        G.add_edges_from(self.get_edges())
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        
        # Draw state nodes
        state_nodes = self.get_nodes_by_type(StateNode)
        nx.draw_networkx_nodes(G, pos, nodelist=state_nodes, node_color="gray", node_size=3000, node_shape="o")
        
        # Draw state nodes
        evidence_nodes = self.get_nodes_by_type(EvidenceNode)
        nx.draw_networkx_nodes(G, pos, nodelist=evidence_nodes, node_color="orange", node_size=3000, node_shape="o")
        
        # Draw action nodes
        action_nodes = self.get_nodes_by_type(ActionNode)
        nx.draw_networkx_nodes(G, pos, nodelist=action_nodes, node_color="tab:blue", node_size=3000, node_shape="s")
        
        # Draw action nodes
        utility_nodes = self.get_nodes_by_type(UtilityNode)
        nx.draw_networkx_nodes(G, pos, nodelist=utility_nodes, node_color="green", node_size=3000, node_shape="d")
        
        # Draw network edges
        nx.draw_networkx_edges(G, pos, node_size=3000)
        
        # Draw node labels
        labels = {n: n for n in self.node_map}
        nx.draw_networkx_labels(G, pos, labels)
        
            
    def is_leaf(self, node_id: Id) -> bool:
        # For a node to be leaf it cant have children
        return len(self.graph[node_id]) == 0
    
    
    def is_root(self, node_id: Id) -> bool:
        # For a node to be root it cant have parents
        parents = []
        for key in self.graph:
            if node_id in self.graph[key]:
                parents.append(key)
        return len(parents) == 0
    
        
    def get_edges(self) -> list[(Id, Id)]:
        edges = []
        for s in self.graph:
            for d in self.graph[s]:
                edges.append((s, d))
        return edges
    
        
    def add_pt(self, node_id: Id, pt: dict[Union[Id, str], int]):
        self.node_map[node_id].add_pt(pt)
        
        
    def get_nodes_by_type(self, node_type: Type(Node)) -> list[Id]:
        return [k for k, v in self.node_map.items() if type(v) is node_type]
    
    
    def get_nodes_by_type_and_time(self, node_type: Type(Node), time: int) -> list[Id]:
        return [node for node in self.get_nodes_by_type(node_type) if self.node_map[node].get_time() == time]
    
    
    def query(self, query: list[Id], evidence: dict[Id, int] = {}, n_samples: int = 1000) -> pd.DataFrame:
        
        # Add value of actions to the nodes
        action_nodes = [k for k in evidence if type(self.node_map[k]) is ActionNode]
        for node in action_nodes:
            self.node_map[node].add_value(evidence[node])
        
        # Create empty sampling dictionary
        sample_dict = {node_id: [] for node_id in self.node_map}
        
        # Create multiple samples
        cur_samples = 0
        while (cur_samples < n_samples):
            
            # Create empty sample
            sample = {}
            queue = [k for k in self.node_map if self.is_root(k)]
            
            # Sample a result from each root node
            while len(queue) != 0:
                
                # Sample from head of queue
                sample[queue[0]] = self.node_map[queue[0]].get_sample(sample)
                
                # Add head's children to queue
                queue += self.graph[queue[0]]
                
                # Remove head from queue
                queue.pop(0)
            
            # Pass sample results to sample_dict if it matches with evidence
            matches = [sample[node_id] == evidence[node_id] for node_id in evidence]
            if all(matches):
                for node_id in sample_dict:
                    sample_dict[node_id].append(sample[node_id])
                cur_samples += 1
                
        # Turn result into probability table
        df = pd.DataFrame(sample_dict)
        df = df.value_counts(normalize=True).to_frame("Prob")
        
        # Group over query variables and sum over all other variables
        df = df.groupby(query).sum().reset_index()
        
        return df
    
    
    def increase_time_step(self):
        # Increase every node time step
        order_key = lambda nid: -nid[1] # Order them in decreasing time order
        node_ids = sorted([nid for nid in self.node_map], key=order_key)
        for (name, t) in node_ids:
            self.node_map[(name, t)].time += 1
            self.node_map[(name, t+1)] = self.node_map.pop((name, t))
            
        # Increase every edge time step
        for (n1, t1) in node_ids:
            for i, (n2, t2) in enumerate(self.graph[(n1, t1)]):
                self.graph[(n1, t1)][i] = (n2, t2+1)
            self.graph[(n1, t1+1)] = self.graph.pop((n1, t1))
            
        # Increase own time clock
        self.time += 1
        
    
    def initialize(self):
        # Get all evidence and reward nodes ids
        init_nodes = self.get_nodes_by_type(EvidenceNode) + self.get_nodes_by_type(UtilityNode)
        
        # Get a single sample from the initial network
        self.node_map[("Action", 0)].add_value(1) # FIX THIS
        sample = self.query(query=init_nodes, n_samples=1)
        sample = {col: int(sample[col]) for col in sample if col != "Prob"}
        self.knowns = sample
        
        # Increase the time for every node and edge
        self.increase_time_step()
        
        # Add A0, E1 and R1 nodes as copies of A1, E2 and R2
        new_nodes = []
        for nid, node in self.node_map.items():
            # Add action nodes
            if type(node) is ActionNode:
                new_nodes.append(ActionNode(node.get_name(), node.get_time()-1, node.get_actions()))
            elif type(node) is EvidenceNode:
                new_nodes.append(EvidenceNode(node.get_name(), node.get_time()-1))
            elif type(node) is UtilityNode:
                new_nodes.append(UtilityNode(node.get_name(), node.get_time()-1))
        self.add_nodes(new_nodes)
        
        # Get node groups
        A1Nodes = self.get_nodes_by_type_and_time(ActionNode, self.time)
        X2Nodes = self.get_nodes_by_type_and_time(StateNode, self.time+1)
        E2Nodes = self.get_nodes_by_type_and_time(EvidenceNode, self.time+1)
        R2Nodes = self.get_nodes_by_type_and_time(UtilityNode, self.time+1)
        
        # Add every edge that's missing
        for (n1, t1) in self.graph:
            # Add Edges (A0, X1) as copy of (A1, X2)
            if (n1, t1) in A1Nodes:
                for (n2, t2) in self.graph[(n1, t1)]:
                    if (n2, t2) in X2Nodes:
                        self.graph[(n1, t1-1)].append((n2, t2-1))
        
            elif (n1, t1) in X2Nodes:
                for (n2, t2) in self.graph[(n1, t1)]:
                    # Add Edges (X1, E1) as copy of (X2, E2)
                    if (n2, t2) in E2Nodes:
                        self.graph[(n1, t1-1)].append((n2, t2-1))
        
                    # Add Edges (X1, R1) as copy of (X2, R2)
                    elif (n2, t2) in R2Nodes:
                        self.graph[(n1, t1-1)].append((n2, t2-1))
        
        # Change X1 cpt to X2 cpt
        for (n, t) in X2Nodes:
            self.node_map[(n, t-1)].pt = self.node_map[(n, t)].pt