import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


"""
x : Input vector(s) (N.., I)
w: weight (.., O, I)
bias: (.., O)
"""
def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    if bias is not None:
        return x @ w.swapaxes(-2,-1) + bias
    return x @ w.swapaxes(-2,-1)


def silu(x: mx.array) -> mx.array:
    pass
