import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.D = dims
        self.HALF_DIM = self.D // 2
        self.L = seq_len
        self.base = base
        self.traditional = traditional
        

        pos = mx.arange(self.L)  # token indices (L,)
        i = mx.arange(self.D // 2)  # component pair indices (D // 2,)
        self.freqs = pos[:,None] * 1.0 / mx.power(self.base, 2.0 * i / self.D)  # (L, D // 2)

        # (L, D // 2)
        self.cosines = mx.cos(self.freqs)
        self.sines = mx.sin(self.freqs)


    """
    x: (N, L, H, D)
    """

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        s = x.shape

        if self.traditional:
            x = x.reshape(*x.shape[:-1], self.D // 2, 2)  # (..., D // 2, 2)
        # else (..., D)
        


        cos, sin = self.cosines, self.sines
        if offset is not None:
            cos = cos[offset,...]
            sin = sin[offset,...]
        else:
            seq_len = x.shape[1] # Note to self: L is the *Maximum* seq length, not necessarily the seq length of the input
            cos = cos[:seq_len,...]
            sin = sin[:seq_len,...]

        if self.traditional:
            cos = cos[None, :, None, :, None]  # (1, L, 1, D//2, 1)
            sin = sin[None, :, None, :, None]
        else:
            cos = cos[None, :, None, :]  # (1, L, 1, D // 2)
            sin = sin[None, :, None, :]


        # A 2D rotation is given by taking a linear combination of a vector and its perpendicular weighted by cos and sin respectively
       

        if self.traditional:
            perp = mx.stack([-1.0 * x[..., 1], x[..., 0]],
                        axis=-1)  # (..., D // 2, 2)
            out = cos * x + sin * perp 
        else:
            x1, x2 =  x[...,:self.HALF_DIM], x[...,self.HALF_DIM:] # (..., D // 2)
            half_one = cos * x1 - sin * x2
            half_two = cos * x2 + sin * x1
            out = mx.concat([half_one,half_two],axis=-1)

        out = out.reshape(*s)

        return out

