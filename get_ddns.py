from src.networks.dn import StateType, ActionType, ObservationType, RewardType
from src.networks.ddn import DDNFactory as DDN
from src.networks.nodes import DiscreteNode


def get_tiger_ddn(base, discount):
    # Create nodes
    s0 = DiscreteNode(("S", 0), node_type=StateType, value_space=list(range(2)))
    a0 = DiscreteNode(("A", 0), node_type=ActionType, value_space=list(range(3)))
    s1 = DiscreteNode(("S", 1), node_type=StateType, value_space=list(range(2)))
    o1 = DiscreteNode(("O", 1), node_type=ObservationType, value_space=list(range(2)))
    r1 = DiscreteNode(("R", 1), node_type=RewardType, value_space=[-10,-1,5])
    # Settings for the POMDP
    nodes = [s0, s1, o1, r1, a0]
    edges = [
        (("S", 0), ("S", 1)),
        (("S", 0), ("R", 1)),
        (("A", 0), ("S", 1)), 
        (("A", 0), ("R", 1)), 
        (("A", 0), ("O", 1)), 
        (("S", 1), ("O", 1)) 
    ]
    # Create the DDN structure
    ddn = DDN(base)(discount=discount)
    ddn.add_nodes(nodes)
    ddn.add_edges(edges)
    # Add CPTs
    data = {
        ("S", 0): [0,1], 
        "Prob": [0.5,0.5]
    }
    ddn.add_pt(("S", 0), data)
    data = {
        ("S", 0): [0,0,0,0,0,0,1,1,1,1,1,1], 
        ("A", 0): [0,0,1,1,2,2,0,0,1,1,2,2], 
        ("S", 1): [0,1,0,1,0,1,0,1,0,1,0,1], 
        "Prob": [1,0,0.5,0.5,0.5,0.5,0,1,0.5,0.5,0.5,0.5]
    }
    ddn.add_pt(("S", 1), data)
    data = {
        ("A", 0): [0,0,0,0,1,1,1,1,2,2,2,2], 
        ("S", 1): [0,0,1,1,0,0,1,1,0,0,1,1], 
        ("O", 1): [0,1,0,1,0,1,0,1,0,1,0,1], 
        "Prob": [0.85,0.15,0.15,0.85,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    }
    ddn.add_pt(("O", 1), data)
    data = {
        ("A", 0): [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2], 
        ("S", 0): [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1], 
        ("R", 1): [-10,-1,5,-10,-1,5,-10,-1,5,-10,-1,5,-10,-1,5,-10,-1,5], 
        "Prob":   [0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,1,0,0]
    }
    ddn.add_pt(("R", 1), data)
    ddn.initialize()
    
    return ddn


def get_robot_ddn(base, discount):
    # Create nodes
    s0 = DiscreteNode(("S", 0), node_type=StateType, value_space=list(range(4)))
    a0 = DiscreteNode(("A", 0), node_type=ActionType, value_space=list(range(4)))
    s1 = DiscreteNode(("S", 1), node_type=StateType, value_space=list(range(4)))
    o1 = DiscreteNode(("O", 1), node_type=ObservationType, value_space=list(range(2)))
    r1 = DiscreteNode(("R", 1), node_type=RewardType, value_space=[-20,-5,-1,10])
    # Settings for the POMDP
    nodes = [s0, s1, o1, r1, a0]
    edges = [
        (("S", 0), ("S", 1)),
        (("S", 0), ("R", 1)), 
        (("A", 0), ("S", 1)),
        (("A", 0), ("R", 1)),
        (("S", 1), ("O", 1))
    ]
    # Create the DDN structure
    ddn = DDN(base)(discount=discount)
    ddn.add_nodes(nodes)
    ddn.add_edges(edges)
    # Add DDN CPTs
    data = {
        ("S", 0): [0,1,2,3], 
        "Prob": [1/3,0,1/3,1/3]
    }
    ddn.add_pt(("S", 0), data)
    data = {
        ("S", 0): [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3], 
        ("A", 0): [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3], 
        ("S", 1): [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3], 
        "Prob":   [0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1]
    }
    ddn.add_pt(("S", 1), data)
    data = { 
        ("S", 1): [0,0,1,1,2,2,3,3], 
        ("O", 1): [0,1,0,1,0,1,0,1], 
        "Prob": [0.9,0.1,0.1,0.9,0.9,0.1,0.9,0.1]
    }
    ddn.add_pt(("O", 1), data)
    data = { 
        ("S", 0): [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
        ("A", 0): [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3], 
        ("R", 1): [-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10], 
        "Prob":   [0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0.3,0,0.7,0.1,0,0,0.9]
    }
    data[("S", 0)] += [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    data[("A", 0)] += [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    data[("R", 1)] += [-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10,-20,-5,-1,10]
    data["Prob"]   += [0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0]
    ddn.add_pt(("R", 1), data)
    ddn.initialize()
    return ddn


def get_gridworld_ddn(base, discount):
    # Create nodes
    s0 = DiscreteNode(("S", 0), node_type=StateType, value_space=list(range(9)))
    a0 = DiscreteNode(("A", 0), node_type=ActionType, value_space=list(range(4)))
    s1 = DiscreteNode(("S", 1), node_type=StateType, value_space=list(range(9)))
    o1 = DiscreteNode(("O", 1), node_type=ObservationType, value_space=list(range(6)))
    r1 = DiscreteNode(("R", 1), node_type=RewardType, value_space=[-2,-1,10])
    # Settings for the POMDP
    nodes = [s0, s1, o1, r1, a0]
    edges = [
        (("S", 0), ("S", 1)),
        (("S", 0), ("R", 1)),
        (("A", 0), ("S", 1)),
        (("A", 0), ("R", 1)),
        (("S", 1), ("O", 1)) 
    ]
    # Create the DDN structure
    ddn = DDN(base)(discount=discount)
    ddn.add_nodes(nodes)
    ddn.add_edges(edges)
    # Add data for node State 0
    data = {
        ("S", 0): [0,1,2,3,4,5,6,7,8], 
        "Prob": [.5,0,0,0,0,0,.5,0,0]
    }
    ddn.add_pt(("S", 0), data)
    # Add data for node State 1
    data = {
        ("S", 0): [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        ("A", 0): [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3],
        ("S", 1): [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
        "Prob":   [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
    }
    data = {
        **data,
        **{
        ("S", 0): [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        ("A", 0): [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3],
        ("S", 1): [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
        "Prob":   [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        }
    }
    data = {
        **data,
        **{
        ("S", 0): [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        ("A", 0): [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3],
        ("S", 1): [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
        "Prob":   [0.5,0,0,0,0,0,0.5,0,0,0.5,0,0,0,0,0,0.5,0,0,0.5,0,0,0,0,0,0.5,0,0,0.5,0,0,0,0,0,0.5,0,0]
        }
    }
    data = {
        **data,
        **{
        ("S", 0): [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
        ("A", 0): [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3],
        ("S", 1): [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
        "Prob":   [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        }
    }
    data = {
        **data,
        **{
        ("S", 0): [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],
        ("A", 0): [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3],
        ("S", 1): [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
        "Prob":   [0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0]
        }
    }
    data = {
        **data,
        **{
        ("S", 0): [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
        ("A", 0): [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3],
        ("S", 1): [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
        "Prob":   [0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        }
    }
    data = {
        **data,
        **{
        ("S", 0): [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
        ("A", 0): [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3],
        ("S", 1): [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
        "Prob":   [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0]
        }
    }
    data = {
        **data,
        **{
        ("S", 0): [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],
        ("A", 0): [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3],
        ("S", 1): [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
        "Prob":   [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        }
    }
    data = {
        **data,
        **{
        ("S", 0): [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
        ("A", 0): [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3],
        ("S", 1): [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
        "Prob":   [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1]
        }
    }
    ddn.add_pt(("S", 1), data)
    # Add data for node Evidence
    data = {
        ("S", 1): [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,8,8,8,8,8,8],
        ("O", 1): [0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5],
        "Prob":   [0.05,0.05,0.05,0.05,0.05,0.75,0.05,0.05,0.05,0.05,0.75,0.05,0.75,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.75,\
                0.05,0.05,0.05,0.05,0.75,0.05,0.05,0.05,0.05,0.75,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.75,0.05,0.05,\
                0.05,0.05,0.75,0.05,0.05,0.05,0.75,0.05,0.05,0.05]
    }   
    ddn.add_pt(("O", 1), data)
    # Add data for node Reward
    data = {
        ("S", 0): [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ("A", 0): [ 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        ("R", 1): [-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10],
        "Prob":   [ 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    }
    data = {
        **data,
        **{
        ("S", 0): [ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ("A", 0): [ 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        ("R", 1): [-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10],
        "Prob":   [ 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
        }
    }
    data = {
        **data,
        **{
        ("S", 0): [ 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ("A", 0): [ 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        ("R", 1): [-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10,-2,-1,10],
        "Prob":   [ 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
        }
    }
    ddn.add_pt(("R", 1), data)
    ddn.initialize()
    
    return ddn