from __future__ import annotations
from itertools import product

def product_dict(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))