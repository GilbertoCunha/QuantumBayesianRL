from __future__ import annotations

Value = float


class Tree:
    
    def __init__(self, value: Value):
        self.value: Value = value
        self.children: list[Tree] = None
        self.parent = None
        
    def get_value(self) -> Value:
        return self.value
    
    def get_children(self) -> list[Tree]:
        return self.children
        
    def add_child(self, child: Tree):
        self.children = [] if self.children is None else self.children
        self.children.append(child)
        
    def add_parent(self, parent: Tree):
        if parent.children is None:
            parent.children = [self]
            self.parent = parent
            
    def get_depth(self) -> int:
        depth = 0
        node = self
        while node.parent is not None:
            node = node.parent
            depth += 1
        return depth