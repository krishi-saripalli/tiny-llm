import mlx.core as mx
from .basics import softmax, linear

"""
Q: (.., L, D) where L is seq_lengh and D is the head_dim
K: (.., L, D)
V: (.., L, D)
mask: (.., L, L)



"""
def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:

    dot_prods = query @ key.swapaxes(-2,-1) # (.., Lq, Lk)
    d_k = key.shape[-1]
    scale = scale if scale is not None else 1.0 / mx.sqrt(d_k)
    dot_prods *= scale


    if mask is not None:
        dot_prods += mask
    
    probs = mx.softmax(dot_prods,axis=-1) # (.., Lq) reduction along keys
    attn = probs @ value # (.., D)
    
    return attn


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        pass

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        pass


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
