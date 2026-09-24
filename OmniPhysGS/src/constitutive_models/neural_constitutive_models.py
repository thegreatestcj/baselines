from typing import *

from functools import partial
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .abstract import Elasticity, Plasticity
import src.constitutive_models.physical_constitutive_models as physical

class PresetElasticity(Elasticity):
    def __init__(
        self,
        elasticity_name: str,
        device: torch.device='cuda',
    ) -> None:
        super().__init__()
        self.elasticity_name = elasticity_name
        self.elasticity = getattr(physical, elasticity_name)().to(device)

    def forward(self, F: Tensor, *args, **kwargs) -> Tensor:
        return self.elasticity(F)
    
    def name(self) -> str:
        return f'[Elasticity] Physical {self.elasticity_name}'
    
class PresetPlasticity(Plasticity):
    def __init__(
        self,
        plasticity_name: str,
        device: torch.device='cuda',
    ) -> None:
        super().__init__()
        self.plasticity_name = plasticity_name
        self.plasticity = getattr(physical, plasticity_name)().to(device)

    def forward(self, F: Tensor, *args, **kwargs) -> Tensor:
        return self.plasticity(F)
    
    def name(self) -> str:
        return f'[Plasticity] Physical {self.plasticity_name}'
     
class _MaskedGrad(torch.autograd.Function):
    """Identity in forward; in backward keeps the gradient only for the rows in `mask` and drops
    non-finite entries (baselines fix). Each expert gets its own masked view of F so that the SVD
    adjoint of an expert evaluated on particles NOT assigned to it (whose stress is multiplied by
    an exact 0 of the hard softmax) can no longer poison the shared F gradient with 0 * inf = NaN
    (e.g. Corotated/StVK evaluated on the exactly isotropic F that SigmaPlasticity produces)."""

    @staticmethod
    def forward(ctx, F: Tensor, mask: Tensor) -> Tensor:
        ctx.save_for_backward(mask)
        return F.view_as(F)

    @staticmethod
    def backward(ctx, grad: Tensor):
        mask, = ctx.saved_tensors
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return grad * mask.view(-1, 1, 1).to(grad.dtype), None


def _mixture(physicals: nn.ModuleList, F: Tensor, category: Tensor) -> Tensor:
    """Hard-softmax mixture of the experts' outputs with per-expert gradient masking (see _MaskedGrad)."""
    max_category = hard_softmax(category, dim=1)
    max_category = max_category.unsqueeze(dim=2).unsqueeze(dim=3)  # num_particles * category_dim * 1 * 1
    assigned = category.argmax(dim=1)
    outs = [p(_MaskedGrad.apply(F, assigned == k)) for k, p in enumerate(physicals)]  # category_dim * num_particles * 3 * 3
    outs = torch.stack(outs, dim=1)  # num_particles * category_dim * 3 * 3
    return (outs * max_category).sum(dim=1)  # num_particles * 3 * 3


def hard_softmax(logits: Tensor, dim: int) -> Tensor:
    y_soft = logits.softmax(dim=dim)
    index = y_soft.argmax(dim=dim, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(dim=dim, index=index, value=1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

def _build_experts(physicals: List[str], device, learnable: bool, learnable_cfg) -> nn.ModuleList:
    """baselines: expert instances; with `learnable` the variants from learnable.py (nn.Parameters),
    otherwise the upstream fixed-buffer classes. Names may repeat (independent instances)."""
    if learnable:
        from .physical_constitutive_models.learnable import make_learnable
        return nn.ModuleList([make_learnable(p, learnable_cfg).to(device) for p in physicals])
    return nn.ModuleList([getattr(physical, p)().to(device) for p in physicals])

class GumbelElasticity(Elasticity):
    def __init__(
        self,
        physicals: List[str],
        device: torch.device='cuda',
        learnable: bool=False,
        learnable_cfg: Optional[dict]=None,
    ) -> None:
        super().__init__()
        self.physical_names = list(physicals)
        self.physicals = _build_experts(physicals, device, learnable, learnable_cfg)
        self.category_dim = len(self.physicals)
        
    def forward(self, F: Tensor, elasticity_category: Tensor) -> Tensor:
        assert elasticity_category.shape[1] == self.category_dim
        # max_category = torch.functional.F.gumbel_softmax(elasticity_category, tau=1.0, dim=1, hard=True) # num_particles * category_dim
        # upstream: possible_stress = [p(F) ...]; stress = (stack * hard_softmax).sum(1)  -> see _mixture
        return _mixture(self.physicals, F, elasticity_category)
    
    def name(self) -> str:
        return f'[Elasticity] Neural GumbelElasticity among {", ".join(self.physical_names)}'

class GumbelPlasticity(Plasticity):
    def __init__(
        self,
        physicals: List[str],
        device: torch.device='cuda',
        learnable: bool=False,
        learnable_cfg: Optional[dict]=None,
    ) -> None:
        super().__init__()
        self.physical_names = list(physicals)
        self.physicals = _build_experts(physicals, device, learnable, learnable_cfg)
        self.category_dim = len(self.physicals)
        
    def forward(self, F: Tensor, plasticity_category: Tensor) -> Tensor:
        assert plasticity_category.shape[1] == self.category_dim
        # max_category = torch.functional.F.gumbel_softmax(plasticity_category, tau=5.0, dim=1, hard=True) # num_particles * category_dim
        return _mixture(self.physicals, F, plasticity_category)
      
    def name(self) -> str:
        return f'[Plasticity] Neural GumbelPlasticity among {", ".join(self.physical_names)}'