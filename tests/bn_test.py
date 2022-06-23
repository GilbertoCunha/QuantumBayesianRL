from numpy.testing import assert_almost_equal
from src.networks.bn import BayesianNetwork
from src.networks.nodes import Node
import pandas as pd
import unittest


class TestBayesianNetwork(unittest.TestCase):

    def setUp(self):
        bn = BayesianNetwork()
        bn.add_nodes([Node("a", [0, 1]), Node("b", [0, 1]), Node("c", [0, 1])])
        self.edges = [("a", "b"), ("b", "c"), ("a", "c")]
        bn.add_edges(self.edges)
        a_dict = {"a": [0, 1], "Prob": [0.2, 0.8]}
        b_dict = {"a": [0, 0, 1, 1], 
                  "b": [0, 1, 0, 1], 
                  "Prob": [0.2, 0.8, 0.3, 0.7]}
        c_dict = {"a": [0, 0, 0, 0, 1, 1, 1, 1], 
                  "b": [0, 0, 1, 1, 0, 0, 1, 1], 
                  "c": [0, 1, 0, 1, 0, 1, 0, 1], 
                  "Prob": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]}
        bn.add_pt("a", a_dict)
        bn.add_pt("b", b_dict)
        bn.add_pt("c", c_dict)
        self.bn = bn
        self.a_dict = a_dict
        self.b_dict = b_dict
        self.c_dict = c_dict

    def test_add_nodes(self):
        self.assertTrue(set(self.bn.get_nodes()) == {"a", "b", "c"})

    def test_add_edges(self):
        self.assertTrue(set(self.bn.get_edges()) == set(self.edges))

    def test_get_parents(self):
        a_parents = set(self.bn.get_parents("a"))
        b_parents = set(self.bn.get_parents("b"))
        c_parents = set(self.bn.get_parents("c"))
        self.assertTrue(a_parents == set())
        self.assertTrue(b_parents == {"a"})
        self.assertTrue(c_parents == {"a", "b"})

    def test_is_root(self):
        self.assertTrue(self.bn.is_root("a"))
        self.assertFalse(self.bn.is_root("b"))
        self.assertFalse(self.bn.is_root("c"))

    def test_is_leaf(self):
        self.assertTrue(self.bn.is_leaf("c"))
        self.assertFalse(self.bn.is_leaf("b"))
        self.assertFalse(self.bn.is_leaf("a"))

    def test_add_pt(self):
        self.assertTrue(pd.DataFrame.equals(
            pd.DataFrame(self.a_dict), self.bn.get_pt("a")))
        self.assertTrue(pd.DataFrame.equals(
            pd.DataFrame(self.b_dict), self.bn.get_pt("b")))
        self.assertTrue(pd.DataFrame.equals(
            pd.DataFrame(self.c_dict), self.bn.get_pt("c")))

    def test_initialize(self):
        self.bn.initialize()
        node_queue = self.bn.get_node_queue()
        prev_nodes = []
        for node in node_queue:
            self.assertTrue(
                all(elem in prev_nodes for elem in self.bn.get_parents(node)))
            prev_nodes.append(node)

    def test_query(self):
        self.bn.initialize()

        # Query the probability table
        a_query = self.bn.query(["a"])
        b_query = self.bn.query(["b"])
        c_query = self.bn.query(["c"])

        # Get pt from the network
        a_pt = self.bn.get_pt("a")
        b_pt = self.bn.get_pt("b").groupby(
            ["b"])[["Prob"]].agg("sum").reset_index()
        c_pt = self.bn.get_pt("c").groupby(
            ["c"])[["Prob"]].agg("sum").reset_index()

        # Normalize pts
        b_pt["Prob"] /= b_pt["Prob"].sum()
        c_pt["Prob"] /= c_pt["Prob"].sum()
        assert_almost_equal(a_query.to_numpy(), a_pt.to_numpy(), decimal=1)
        assert_almost_equal(b_query.to_numpy(), b_pt.to_numpy(), decimal=1)
        assert_almost_equal(c_query.to_numpy(), c_pt.to_numpy(), decimal=1)


if __name__ == '__main__':
    unittest.main()
