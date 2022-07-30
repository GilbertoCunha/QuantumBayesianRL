from __future__ import annotations
from typing import Hashable, Any


class Tree:
    
    def __init__(self, attributes: dict = None):
        self.attributes = {} if attributes is None else attributes
        self.children: list[Tree] = []
        self.parent: Tree = None
    
    def get_children(self) -> list[Tree]:
        return self.children
        
    def get_attribute(self, attribute: Hashable) -> Any:
        return self.attributes[attribute]
        
    def add_attribute(self, attribute: Hashable, value: Any):
        self.attributes[attribute] = value
        
    def add_attributes(self, attributes: dict):
        for attribute in attributes:
            self.add_attribute(attribute, attributes[attribute])
        
    def add_child(self, child: Tree):
        self.children.append(child)
        child.parent = self
            
    def get_depth(self) -> int:
        depth = 0
        node = self
        while node.parent is not None:
            node = node.parent
            depth += 1
        return depth