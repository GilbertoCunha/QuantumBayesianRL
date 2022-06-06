from __future__ import annotations


class TreeNode:
    def __init__(self, value: int, depth: int):
        self.value: int = value
        self.depth: int = depth
        self.parent: TreeNode = None
        self.children: TreeNode = []
            
    def get_value(self):
        return self.value
    
    def get_depth(self):
        return self.depth
    
    def get_parent(self):
        return self.parent
            
    def get_children(self):
        return self.children
        
    def add_parent(self, parent: TreeNode):
        self.parent = parent
        
    def add_children(self, child: TreeNode):
        self.children.append(child)
        
        
class TreeBeliefNode(TreeNode):
    pass


class TreeObservationNode(TreeNode):
    pass