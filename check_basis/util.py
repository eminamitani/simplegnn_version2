import torch
import torch.nn as nn


# ---- Radial 基底の抽象クラス ----
class RadialBasis(nn.Module):
    def __init__(self, num_basis: int, rc: float):
        super().__init__()
        self.num_basis = num_basis
        self.rc = rc
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# ---- 具体基底：Sinc ----
class SincBasis(RadialBasis):
    def __init__(self, num_basis: int, rc: float):
        super().__init__(num_basis, rc)
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # (E, K)
        return sinc_expansion(r, self.num_basis, self.rc)

# ---- 具体基底：Chebyshev ----
class ChebyshevBasis(RadialBasis):
    def __init__(self, num_basis: int, rc: float, normalize: bool = True):
        super().__init__(num_basis, rc)
        self.normalize = normalize
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return chebyshev_basis(r, self.num_basis, self.rc, self.normalize)


# ---- Envelope 抽象クラス ----
class Envelope(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# ---- 具体エンベロープ：cosine ----
class CosineCutoff(Envelope):
    def __init__(self, rc: float):
        super().__init__(rc)
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return cosine_cutoff(r, self.rc).unsqueeze(-1)  # (E,1)

# ---- 具体エンベロープ：DimeNet++風 polynomial ----
class PolyEnvelope(Envelope):
    def __init__(self, rc: float):
        super().__init__(rc)
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return envelope_poly(r, self.rc).unsqueeze(-1) # (E,1)

# ---- 何も掛けない（デバッグ用）----
class IdentityEnvelope(Envelope):
    def __init__(self, rc: float):
        super().__init__(rc)
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(r).unsqueeze(-1)
    


# ============================================================
# Factory
# ============================================================

def make_radial(name: str, num_basis: int, rc: float, **kw) -> RadialBasis:
    name = name.lower()
    if name in ("sinc", "sinc_expansion"):
        return SincBasis(num_basis, rc)
    elif name in ("cheby", "chebyshev"):
        return ChebyshevBasis(num_basis, rc, kw.get("normalize", True))

    else:
        raise ValueError(f"Unknown radial basis: {name}")


def make_envelope(name: str, rc: float) -> Envelope:
    name = name.lower()
    if name in ("cos", "cosine", "cosine_cutoff"):
        return CosineCutoff(rc)
    elif name in ("poly", "envelope_poly", "dimenet"):
        return PolyEnvelope(rc)
    elif name in ("none", "identity"):
        return IdentityEnvelope(rc)
    else:
        raise ValueError(f"Unknown envelope: {name}")

'''
radial basis function  
'''
@torch.jit.script
def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float, eps: float = 1e-12):
    n = torch.arange(edge_size, device=edge_dist.device, dtype=edge_dist.dtype) + 1.0
    x = edge_dist.unsqueeze(-1) * n * torch.pi / cutoff
    denom = edge_dist.unsqueeze(-1).clamp_min(eps)
    return torch.sin(x) / denom

@torch.jit.script
def chebyshev_basis(r: torch.Tensor, num_basis: int, rc: float, normalize: bool = False) -> torch.Tensor:
    """
    r: (E,) 距離
    num_basis: 基底数 K
    rc: カットオフ
    return: (E, K) で [T_0(x), T_1(x), ..., T_{K-1}(x)] * cutoff を返す
    """
    # x in [-1, 1]
    x = 2.0 * r / rc - 1.0
    x = torch.clamp(x, -1.0, 1.0)

    E = r.shape[0]
    K = num_basis
    B = torch.empty((E, K), dtype=r.dtype, device=r.device)

    # 再帰で安定＆自動微分OK（arccosは使わない）
    T0 = torch.ones_like(x)
    B[:, 0] = T0
    if K > 1:
        T1 = x
        B[:, 1] = T1
        for n in range(2, K):
            T2 = 2.0 * x * T1 - T0
            B[:, n] = T2
            T0, T1 = T1, T2

    if normalize:
        # Chebyshev(T) の標準直交重み (1/sqrt(1-x^2)) に対する規格化を意識したスケール
        # 実用上は n>0 を sqrt(2) 倍することが多い
        scale = torch.ones(K, dtype=r.dtype, device=r.device)
        if K > 1:
            scale[1:] = scale[1:] * (2.0 ** 0.5)
        B = B * scale

    return B 


'''
cutoff functions
'''

@torch.jit.script
def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """
    condition = (edge_dist < cutoff).to(edge_dist.device)  
    return torch.where(
        condition,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype), 
    )

@torch.jit.script
def envelope_poly(r: torch.Tensor, rc: float) -> torch.Tensor:
    # DimeNet++風: t = r/rc,  a0 + a1 t + a2 t^2 + a3 t^3 で
    # t=1 で値/1階/2階微分=0 になる係数（簡易版）
    t = torch.clamp(r / rc, 0.0, 1.0)
    a0, a1, a2, a3 = 1.0, -3.0, 3.0, -1.0  # (1 - t)^3 に相当
    env = a0 + a1*t + a2*t*t + a3*t*t*t
    return env * (r < rc).to(r.dtype)