from __future__ import annotations
from typing import Hashable, Any

Value = float


class Tree:
    
    def __init__(self, value: Value, attributes: dict = None):
        self.value = value
        self.attributes = {} if attributes is None else attributes
        self.children: list[Tree] = None
        self.parent: Tree = None
        
    def get_value(self) -> Value:
        return self.value
    
    def get_children(self) -> list[Tree]:
        return self.children
        
    def get_attribute(self, attribute: Hashable) -> Any:
        return self.attributes[attribute]
        
    def add_value(self, value: Value):
        self.value = value
        
    def add_attribute(self, attribute: Hashable, value: Any):
        self.attributes[attribute] = value
        
    def add_attributes(self, attributes: dict):
        for attribute in self.attributes:
            self.add_attribute(attribute, attributes[attribute])
        
    def add_child(self, child: Tree):
        self.children = [] if self.children is None else self.children
        self.children.append(child)
        child.add_parent(self)
        
    def add_parent(self, parent: Tree):
        parent.children = [] if parent.children is None else parent.children
        parent.children.append(self)
        self.parent = parent
            
    def get_depth(self) -> int:
        depth = 0
        node = self
        while node.parent is not None:
            node = node.parent
            depth += 1
        return depth