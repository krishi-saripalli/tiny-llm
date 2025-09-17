import mlx.core as mx
from .basics import softmax, linear

"""
Q: (.., L, D) where L is seq_length and D is the head_dim
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
    
    probs = mx.softmax(dot_prods,axis=-1) # (..Lq, Lk) we're normalizing along the key dimension 
    attn = probs @ value # (.., Lq, D)
    
    return attn


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array, # (H*D, E)
        wk: mx.array,  # (H*D, E)
        wv: mx.array,  # (H*D, E)
        wo: mx.array,  # (E, H*D)
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    """
    query: (N,L,E)
    key: (N,L,E)
    value: (N,L,E)
    """
    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:

        # D is head dimension, a chunk of the total hidden dimension E
        

        # Apply individual projections to achieve (N, L, H*D)
        q_proj = linear(query,self.wq)
        k_proj = linear(key,self.wk)
        v_proj = linear(value,self.wv)

        # Reshape so that we get (N, H, L, D)
        q_proj = q_proj.reshape(*q_proj.shape[:-1],self.num_heads,self.head_dim).swapaxes(-3,-2)
        k_proj = k_proj.reshape(*k_proj.shape[:-1],self.num_heads,self.head_dim).swapaxes(-3,-2)
        v_proj = v_proj.reshape(*v_proj.shape[:-1],self.num_heads,self.head_dim).swapaxes(-3,-2)

        # Calculate per head attention (N,H,L,D)
        sdpa = scaled_dot_product_attention_simple(q_proj,k_proj,v_proj,mask=mask)

       # Merge heads back together (N, L, H*D) to get ready for output projection (E, H*D)
        sdpa_swapped = sdpa.swapaxes(-3,-2)
        sdpa = sdpa_swapped.reshape(*sdpa_swapped.shape[:-2],self.num_heads*self.head_dim)

        # output should be (N, L, E) i.e transformed embedding for each token in the sequence
        return linear(sdpa,self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass

'''
    N.. is zero or more dimensions for batches
    H_q is the number of query heads
    H is the number of key/value heads (H_q must be divisible by H)
    L is the query sequence length
    S is the key/value sequence length
    D is the head dimension

    query: N.. x H_q x L x D
    key: N.. x H x S x D
    value: N.. x H x S x D
    mask: N.. x H_q x L x S
    output: N.. x H_q x L x D
'''
def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    
    H_q = query.shape[-3]
    H_kv = key.shape[-3]
    G = H_q // H_kv

    query_grouped = query.reshape(*query.shape[:-3],H_kv, G,*query.shape[-2:]) # (N, ..., H_kv, G L, D)
    key_grouped = mx.expand_dims(key,axis=-3).swapaxes(-2,-1) # (N,...,1, H_kv, D, S)
    dot_prods = query_grouped @ key_grouped  # (.., L, S)
    d_k = key.shape[-1]
    scale = scale if scale is not None else 1.0 / mx.sqrt(d_k)
    dot_prods *= scale


    if mask is not None:
        dot_prods += mask.reshape(*mask.shape[:-3],H_kv, G,*mask.shape[-2:])
    
    probs = mx.softmax(dot_prods,axis=-1) # (..L, S) we're normalizing along the key dimension 
    attn = probs @ mx.expand_dims(value,axis=-3) # (.., H_kv, G  L, D)
    attn = attn.reshape(*attn.shape[:-4],H_q,*attn.shape[-2:])
    
    return attn
    


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
