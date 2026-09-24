from typing import *

import src.constitutive_models.physical_constitutive_models as physical
from .neural_constitutive_models import *

def get_elasticity(elasticity_name: str, physicals: Optional[List[str]]=None, device: str='cuda',
                   learnable: bool=False, learnable_cfg: Optional[dict]=None):
    """baselines: `learnable=True` resolves every expert to its learnable variant
    (physical_constitutive_models/learnable.py) with parameters initialised from `learnable_cfg`."""
    if elasticity_name == 'neural':
        return GumbelElasticity(
            physicals=physicals,
            device=device,
            learnable=learnable,
            learnable_cfg=learnable_cfg,
        )
    return PresetElasticity(elasticity_name, device=device)

def get_plasticity(plasticity_name: str, physicals: Optional[List[str]]=None, device: str='cuda',
                   learnable: bool=False, learnable_cfg: Optional[dict]=None):
    if plasticity_name == 'neural':
        return GumbelPlasticity(
            physicals=physicals,
            device=device,
            learnable=learnable,
            learnable_cfg=learnable_cfg,
        )
    return PresetPlasticity(plasticity_name, device=device)

