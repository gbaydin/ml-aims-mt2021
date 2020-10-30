import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None # this will be called on the backward pass
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError

    def relu(self):
        raise NotImplementedError

    def sigmoid(self):
        raise NotImplementedError

    def cos(self):
        raise NotImplementedError

    def sin(self):
        raise NotImplementedError

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            # this needs to fill topo with the nodes in a topological ordering of the graph
            # so we can visit them and call their _backward() in the correct order

            raise NotImplementedError
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        raise NotImplementedError

    def __radd__(self, other): # other + self
        raise NotImplementedError

    def __sub__(self, other): # self - other
        raise NotImplementedError

    def __rsub__(self, other): # other - self
        raise NotImplementedError

    def __rmul__(self, other): # other * self
        raise NotImplementedError

    def __truediv__(self, other): # self / other
        raise NotImplementedError

    def __rtruediv__(self, other): # other / self
        raise NotImplementedError

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"