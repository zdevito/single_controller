from typing import List, Sequence, Union, Tuple, overload
from dataclasses import dataclass
from itertools import product
import math
import random

@dataclass(frozen=True)
class NDSlice:
    """
    A multidimension version of Python's slice object.
    Think of a single dimensional slice as a compression of any sequence of integers 
    representable as (start, stop, stride). A NDSlice is also a compression of a sequence
    of integers except we can compress a sequence of integers that can be represented by
    multiple dimenisons of size/stride info with a single starting offset. This representation
    is good for compressing the set of workers we need to send commands to since they are often
    a slice of the overall ranks of processes.

    Like a slice, functions __getitem__ and index work on the object as if it were a flat sequence
    of integers even if self.ndim > 1. Additional functions nditem() and coordinates() are provided
    to look up the multi-dimension indices of values for easier manipulation of the slice.
    """
    offset: int
    sizes: List[int]
    strides: List[int]

    @property
    def ndim(self) -> int:
        return len(self.sizes)

    def __post_init__(self):
        if len(self.sizes) != len(self.strides):
            raise ValueError("sizes and strides must have the same length")
        prev_stride = None
        total = 1
        for stride, size in sorted(zip(self.strides, self.sizes)):
            if prev_stride is not None:
                if stride % prev_stride != 0:
                    raise ValueError("NDSlice must be rectangularly shaped.")
                if stride == prev_stride:
                    raise ValueError("Strides must be unique.")
            if size <= 0:
                raise ValueError("Slice sizes must be positive.")
            if total > stride:
                raise ValueError("Stride must be positive, and larger than size of previous space.")
            total = stride * size
            prev_stride = stride

    def __iter__(self):
        # return the flat list of values that this multidimensional slice
        # would produce
        for loc in product(*(range(s) for s in self.sizes)):
            yield self.offset + sum(i*s for i, s in zip(loc, self.strides))

    def union(self, other: 'NDSlice') -> List['NDSlice']:
        raise NotImplementedError()
    
    def contains_any(self, start: int, end: int) -> bool:
        # does this slice contain any of the elements in [start, end)
        # will be used to figure out who to broadcast to.
        raise NotImplementedError()
    
    def index(self, value: int) -> int:
        # return _flat_ index  where self[index] == value,
        # or raise ValueError if not found
        coords = self.coordinates(value)
        stride = 1
        result = 0
        for idx, size in zip(reversed(coords), reversed(self.sizes)):
            result += idx * stride
            stride *= size

        return result

    def coordinates(self, value: int) -> List[int]:
        # return the multidimension coordinates of where
        # value would occur, a list of length self.ndim
        pos = value - self.offset
        if pos < 0:
            raise ValueError(f"value {value} not in NDSlice {self} ()")
        result = [0] * len(self.sizes)
        sorted_info = sorted(zip(self.strides, enumerate(self.sizes)))
        for stride, (i, size) in reversed(list(sorted_info)):
            index, pos = divmod(pos, stride)
            if index >= size:
                raise ValueError(f"value {value} not in NDSlice {self}")
            result[i] = index
        if pos != 0:
            raise ValueError(f"value {value} not in NDSlice {self}")
        return result

    def nditem(self, coordinates: Sequence[int]):
        if len(coordinates) != self.ndim:
            raise IndexError("index has wrong number of dimensions")
        for i, size in zip(coordinates, self.sizes):
            if i < 0 or i >= size:
                raise IndexError("index out of range")
        return self.offset + sum(i*s for i, s in zip(coordinates, self.strides))

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, index: slice) -> Tuple[int]:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            N = math.prod(self.sizes)
            return tuple(self[i] for i in range(*index.indices(N)))

        # return the value as if we did tuple(self)[index],
        # but do so in O(1) time
        value = self.offset
        rest = index
        N = 1
        for size, stride in zip(reversed(self.sizes), reversed(self.strides)):
            N *= size
            value += (rest % size) * stride
            rest //= size
        if index < 0 or index >= N:
            raise IndexError("index out of range")
        return value
