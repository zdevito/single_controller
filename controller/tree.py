from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map

def flatten(tree, cond):
    r, spec = tree_flatten(tree)
    def unflatten(n):
        n_it = iter(n)
        return tree_unflatten([next(n_it) if cond(e) else e for e in r], spec)
    return [e for e in r if cond(e)], unflatten


__all__ = [
    'flatten', 
    'tree_map',
]
