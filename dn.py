from bn import BayesianNetwork, BinaryNode
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import itertools

class BinaryActionNode(BinaryNode):
    
    def __init__(self, name, actions):
        super().__init__(name)
        self.actions = actions
        
    def add_value(self, value):
        self.pt = pd.DataFrame({self.name: [value, 1-value], "Prob": [1, 0]})


class BinaryUtilityNode(BinaryNode):
    pass


class DecisionNetwork(BayesianNetwork):
    
    def __init__(self):
        self.graph = {}
        self.nodes = []
        self.edges = []
        
    def add_action_node(self, name, actions):
        if name not in self.graph:
            self.graph[name] = BinaryActionNode(name, actions)
            self.nodes.append(name)
            
    def add_action_nodes(self, name_action_dict):
        for name in name_action_dict:
            self.add_action_node(name, name_action_dict[name])
            
    def add_utility_node(self, name):
        if name not in self.graph:
            self.graph[name] = BinaryUtilityNode(name)
            self.nodes.append(name)
            
    def add_utility_nodes(self, names):
        for name in names:
            self.add_action_node(name)
            
    def draw(self):
        # Create nx graph
        G = nx.DiGraph(directed=True)
        G.add_edges_from(self.edges)
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        
        # Draw regular nodes
        nodes = self.get_nodes()
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color="orange", node_size=3000, node_shape="o")
        
        # Draw action nodes
        action_nodes = self.get_action_nodes()
        nx.draw_networkx_nodes(G, pos, nodelist=action_nodes, node_color="blue", node_size=3000, node_shape="s")
        
        # Draw action nodes
        utility_nodes = self.get_utility_nodes()
        nx.draw_networkx_nodes(G, pos, nodelist=utility_nodes, node_color="green", node_size=3000, node_shape="d")
        
        # Draw network edges
        nx.draw_networkx_edges(G, pos)
        
        # Draw node labels
        labels = {n: n for n in self.nodes}
        nx.draw_networkx_labels(G, pos, labels)
        plt.show()
        
    @staticmethod
    def bitGen(n):
        return [''.join(i) for i in itertools.product('01', repeat=n)]
        
    def get_action_nodes(self):
        r = [n for n in self.nodes if type(self.graph[n]) is BinaryActionNode]
        return r
    
    def get_utility_nodes(self):
        r = [n for n in self.nodes if type(self.graph[n]) is BinaryUtilityNode]
        return r
    
    def get_nodes(self):
        r = [n for n in self.nodes if type(self.graph[n]) is BinaryNode]
        return r
    
    def query_decision(self, query, evidence, n_samples=1000):
        # Get all action nodes
        action_nodes = self.get_action_nodes()
        
        # Get all actions for all the action nodes
        action_space = {}
        for a in action_nodes:
            action_space[a] = self.graph[a].actions
        
        # Create a list of all possible actions to be taken
        keys, values = zip(*action_space.items())
        action_space = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Iterate each set of actions in action space
        results = []
        for actions in action_space:
            
            # Set the actions of the action nodes to the current set of actions
            for action_node in action_nodes:
                self.graph[action_node].add_value(actions[action_node])
                
            # Perform query
            df = self.query(query=[query], evidence=evidence, n_samples=n_samples)
            
            # Get expected utility
            eu = float((df[query] * df["Prob"]).sum())
            results.append((actions, eu))
            
        # Get the result with the maximum expected utility
        r = max(results, key=lambda x: x[1])[0]
        
        return r
        