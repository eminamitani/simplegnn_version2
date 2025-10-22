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

# ---- 具体基底：Gaussian RBF ----
class GaussianRBF(RadialBasis):
    def __init__(self, num_basis: int, rc: float, start=0.0):
        super().__init__(num_basis, rc)
        self.register_buffer("offsets", torch.linspace(start, rc, num_basis))
        self.register_buffer("widths", torch.full((num_basis,), rc / num_basis))

    def forward(self, distances):
        return gaussian_rbf(distances, self.offsets, self.widths)

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

# ---- 具体エンベロープ：高次多項式 ----
class HighOrderPolyCutoff(torch.nn.Module):
    def __init__(self, rc: float, order: int = 3):
        super().__init__()
        if order < 1 or order > 5:
            raise ValueError("order must be 1..5")
        self.rc = rc
        self.order = order

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        t = (r / self.rc).clamp(0.0, 1.0)
        s = smoothstep_poly(t, self.order)           # S_m(t) in [0,1]
        fc = (1.0 - s) * (r < self.rc).to(r.dtype)   # f_cut = 1 - S, outside=0
        return fc.unsqueeze(-1)
    
# ---- 具体エンベロープ：PolyGamma ----
class PolyGammaEnvelope(Envelope):
    def __init__(self, rc: float, gamma: float):
        super().__init__(rc)
        self.gamma = gamma
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return envelope_poly_gamma(r, self.rc, self.gamma).unsqueeze(-1) # (E,1)

class FractionalEnvelope(Envelope):
    def __init__(self, rc: float, h: float):
        super().__init__(rc)
        self.h = h
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return fractional_cutoff(r, self.rc, self.h).unsqueeze(-1) # (E,1)

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
    elif name in ("gauss", "gaussian", "gaussian_rbf"):
        return GaussianRBF(num_basis, rc, kw.get("start", 0.0))
    else:
        raise ValueError(f"Unknown radial basis: {name}")


def make_envelope(name: str, rc: float, **kw) -> Envelope:
    name = name.lower()
    if name in ("cos", "cosine", "cosine_cutoff"):
        return CosineCutoff(rc)
    elif name in ("poly", "envelope_poly", "dimenet"):
        return PolyEnvelope(rc)
    elif name in ("polygamma", "poly_gamma"):
        return PolyGammaEnvelope(rc, gamma=kw.get("gamma", 2))
    elif name in ("poly_high", "highorder", "smoothstep", "smootherstep"):
        return HighOrderPolyCutoff(rc, order=kw.get("order", 3))
    elif name in ("fractional", "fractional_cutoff"):
        return FractionalEnvelope(rc, h=kw.get("h", 0.5))
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
    Chebyshev 多項式基底
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.014112 
    https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.4.040601


    r: (E,) 距離
    num_basis: 基底数 K
    rc: カットオフ
    return: (E, K) で [T_0(x), T_1(x), ..., T_{K-1}(x)] を返す
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

def gaussian_rbf(inputs, offsets, widths):
    coeff = -0.5 / widths**2
    diff = inputs[..., None] - offsets
    return torch.exp(coeff * diff**2)

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


@torch.jit.script
def smoothstep_poly(t: torch.Tensor, order: int) -> torch.Tensor:
    """
    S_order(t) on t in [0,1]. 係数は TorchScript 内で if/elif により定義。
    order: 1..5 を想定
    """
    # 係数を Torch テンソルで作る（dtype/device を t に合わせる）
    if order == 1:
        coeffs = torch.tensor([0.0, 1.0], dtype=t.dtype, device=t.device)  # t
    elif order == 2:
        coeffs = torch.tensor([0.0, 0.0, 3.0, -2.0], dtype=t.dtype, device=t.device)  # 3t^2 - 2t^3
    elif order == 3:
        coeffs = torch.tensor([0.0, 0.0, 0.0, 10.0, -15.0, 6.0], dtype=t.dtype, device=t.device)  # 10t^3 -15t^4 +6t^5
    elif order == 4:
        coeffs = torch.tensor([0.0, 0.0, 0.0, 0.0, 35.0, -84.0, 70.0, -20.0],
                              dtype=t.dtype, device=t.device)  # 35t^4 -84t^5 +70t^6 -20t^7
    elif order == 5:
        coeffs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 126.0, -420.0, 540.0, -315.0, 70.0],
                              dtype=t.dtype, device=t.device)  # 126t^5 -420t^6 + ...
    else:
        # TorchScriptでは ValueError より RuntimeError の方が扱いやすい
        raise RuntimeError("smoothstep_poly: order must be in {1,2,3,4,5}")

    # Horner 法で多項式評価： y = (...((c_n)*t + c_{n-1})*t + ... + c_0)
    y = torch.zeros_like(t)
    # 末尾（最高次）から前に向かって
    for i in range(int(coeffs.numel()) - 1, -1, -1):
        y = y * t + coeffs[i]
    return y

@torch.jit.script
def envelope_poly_gamma(r: torch.Tensor, rc: float, gamma: float) -> torch.Tensor:
    '''
    Cosine cutoffではカットオフ半径付近であまりにもスムーズにしてしまい、
    多体の相互作用を捉えにくくなる場合がある。
    https://www.sciencedirect.com/science/article/pii/S0010465516301266?via%3Dihub
    '''
    t = torch.clamp(r / rc, 0.0, 1.0)
    env=1+gamma*t**(gamma+1)-(1+gamma)*t**gamma
    return env * (r < rc).to(r.dtype)

@torch.jit.script
def fractional_cutoff(r: torch.Tensor, rc: float, h: float) -> torch.Tensor:
    '''
    Fractional Cutoff function
    https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.7.063605
    '''
    x= (r-rc)/h
    env = x*x / (1 + x*x)
    return env * (r < rc).to(r.dtype)