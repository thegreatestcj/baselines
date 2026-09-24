"""Learnable variants of the expert constitutive models (baselines addition).

Upstream experts carry fixed buffers (log_E = log 2e6, nu = 0.4, sigma_y = 1e3, friction_angle = 25,
cohesion = 0). For system identification we make them `nn.Parameter`s:

  * E            world Pa, parametrised as log E, clamped to `E_range`; the module receives the
                 world->sim length scale `s` through `set_length_scale(s)` and feeds E_sim = E * s^2
                 to the physics (stresses scale as s^2 when lengths scale by s and time/density are
                 unchanged).
  * nu           sigmoid-parametrised to stay inside (0, nu_max=0.49).
  * sigma_y      world Pa (von Mises yield stress), log-parametrised, sigma_y_sim = sigma_y * s^2.
  * friction_angle (deg) and cohesion (log-strain offset, dimensionless), clamped.

`make_learnable(name, cfg)` returns the learnable subclass of the upstream class `name`; the expert
lists in a config may repeat an entry (independent parameters each). `world_params()` reports the
current values in world units.
"""
import math
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from ..abstract import Elasticity, Plasticity
from . import elasticity as E_
from . import plasticity as P_

NU_MAX = 0.49


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class _LearnableMixin:
    """Parameter bookkeeping shared by all learnable experts. Must be initialised after Material.__init__."""

    def _init_params(self, cfg: dict, with_E=True, with_nu=True, with_sigma_y=False, with_dp=False):
        cfg = cfg or {}
        self.register_buffer("scale2", torch.tensor(1.0))  # s^2, set by set_length_scale
        self.param_names = []
        if with_E:
            lo, hi = cfg.get("E_range", [1e3, 1e7])
            self.register_buffer("log_E_min", torch.tensor(math.log(lo)))
            self.register_buffer("log_E_max", torch.tensor(math.log(hi)))
            self.log_E_world = nn.Parameter(torch.tensor([math.log(float(cfg.get("init_E", 1e5)))]))
            self.param_names.append("E")
        if with_nu:
            self.nu_raw = nn.Parameter(torch.tensor([_logit(float(cfg.get("init_nu", 0.3)) / NU_MAX)]))
            self.param_names.append("nu")
        if with_sigma_y:
            lo, hi = cfg.get("sigma_y_range", [1e1, 1e6])
            self.register_buffer("log_sy_min", torch.tensor(math.log(lo)))
            self.register_buffer("log_sy_max", torch.tensor(math.log(hi)))
            self.log_sigma_y_world = nn.Parameter(torch.tensor([math.log(float(cfg.get("init_sigma_y", 1e3)))]))
            self.param_names.append("sigma_y")
        if with_dp:
            lo, hi = cfg.get("friction_angle_range", [5.0, 60.0])
            self.register_buffer("fa_min", torch.tensor(float(lo)))
            self.register_buffer("fa_max", torch.tensor(float(hi)))
            self.friction_angle_raw = nn.Parameter(torch.tensor([float(cfg.get("init_friction_angle", 25.0))]))
            lo, hi = cfg.get("cohesion_range", [0.0, 0.1])
            self.register_buffer("coh_min", torch.tensor(float(lo)))
            self.register_buffer("coh_max", torch.tensor(float(hi)))
            self.cohesion_raw = nn.Parameter(torch.tensor([float(cfg.get("init_cohesion", 0.0))]))
            self.param_names += ["friction_angle", "cohesion"]

    def set_length_scale(self, s: float):
        self.scale2.fill_(float(s) ** 2)

    @torch.no_grad()
    def project_(self):
        """Clamp the raw parameters into their ranges after an optimizer step (the forward clamps
        would otherwise leave a parameter stranded outside its range with zero gradient)."""
        if hasattr(self, "log_E_world"):
            self.log_E_world.clamp_(self.log_E_min, self.log_E_max)
        if hasattr(self, "log_sigma_y_world"):
            self.log_sigma_y_world.clamp_(self.log_sy_min, self.log_sy_max)
        if hasattr(self, "friction_angle_raw"):
            self.friction_angle_raw.clamp_(self.fa_min, self.fa_max)
        if hasattr(self, "cohesion_raw"):
            self.cohesion_raw.clamp_(self.coh_min, self.coh_max)

    # world-unit accessors
    def E_world(self) -> Tensor:
        return torch.clamp(self.log_E_world, self.log_E_min, self.log_E_max).exp()

    def nu(self) -> Tensor:
        return NU_MAX * torch.sigmoid(self.nu_raw)

    def sigma_y_world(self) -> Tensor:
        return torch.clamp(self.log_sigma_y_world, self.log_sy_min, self.log_sy_max).exp()

    def friction_angle(self) -> Tensor:
        return torch.clamp(self.friction_angle_raw, self.fa_min, self.fa_max)

    def cohesion(self) -> Tensor:
        return torch.clamp(self.cohesion_raw, self.coh_min, self.coh_max)

    # sim-unit values used by the physics
    def log_E_sim(self) -> Tensor:
        return torch.clamp(self.log_E_world, self.log_E_min, self.log_E_max) + self.scale2.log()

    def world_params(self) -> Dict[str, float]:
        out = {}
        for n in self.param_names:
            fn = {"E": self.E_world, "nu": self.nu, "sigma_y": self.sigma_y_world,
                  "friction_angle": self.friction_angle, "cohesion": self.cohesion}[n]
            out[n] = float(fn().detach().reshape(-1)[0])
        return out

    def set_world_params(self, **kw):
        """Overwrite parameters from world-unit values (e.g. GT reference rollouts)."""
        with torch.no_grad():
            if "E" in kw and hasattr(self, "log_E_world"):
                self.log_E_world.fill_(math.log(float(kw["E"])))
            if "nu" in kw and hasattr(self, "nu_raw"):
                self.nu_raw.fill_(_logit(float(kw["nu"]) / NU_MAX))
            if "sigma_y" in kw and hasattr(self, "log_sigma_y_world"):
                self.log_sigma_y_world.fill_(math.log(float(kw["sigma_y"])))
            if "friction_angle" in kw and hasattr(self, "friction_angle_raw"):
                self.friction_angle_raw.fill_(float(kw["friction_angle"]))
            if "cohesion" in kw and hasattr(self, "cohesion_raw"):
                self.cohesion_raw.fill_(float(kw["cohesion"]))


# ----------------------------------------------------------------------------- elasticity

def _learnable_elasticity(base):
    class Learnable(_LearnableMixin, base):
        def __init__(self, cfg: dict = None) -> None:
            Elasticity.__init__(self)  # skip the upstream buffers
            self._init_params(cfg)

        def forward(self, F: Tensor, log_E=None, nu=None) -> Tensor:
            return base.forward(self, F, log_E=self.log_E_sim(), nu=self.nu())

    Learnable.__name__ = "Learnable" + base.__name__
    Learnable.__qualname__ = Learnable.__name__
    return Learnable


LearnableCorotatedElasticity = _learnable_elasticity(E_.CorotatedElasticity)
LearnableStVKElasticity = _learnable_elasticity(E_.StVKElasticity)
LearnableSigmaElasticity = _learnable_elasticity(E_.SigmaElasticity)
LearnableVolumeElasticity = _learnable_elasticity(E_.VolumeElasticity)
LearnableFluidElasticity = _learnable_elasticity(E_.FluidElasticity)


# ----------------------------------------------------------------------------- plasticity

class LearnableIdentityPlasticity(_LearnableMixin, P_.IdentityPlasticity):
    def __init__(self, cfg: dict = None) -> None:
        Plasticity.__init__(self)
        self._init_params(cfg, with_E=False, with_nu=False)


class LearnableSigmaPlasticity(_LearnableMixin, P_.SigmaPlasticity):
    def __init__(self, cfg: dict = None) -> None:
        Plasticity.__init__(self)
        self._init_params(cfg, with_E=False, with_nu=False)


class LearnableVonMisesPlasticity(_LearnableMixin, P_.VonMisesPlasticity):
    """Upstream forward with sigma_y, E, nu taken from the parameters (return mapping in sim units)."""

    def __init__(self, cfg: dict = None) -> None:
        Plasticity.__init__(self)
        self._init_params(cfg, with_sigma_y=True)

    def forward(self, F: Tensor, log_E=None, nu=None) -> Tensor:
        E = self.log_E_sim().exp()
        nu = self.nu()
        sigma_y = self.sigma_y_world() * self.scale2
        mu = E / (2 * (1 + nu))
        mu = mu.reshape(-1, 1)
        U, sigma, Vh = self.svd(F)
        sigma = torch.clamp_min(sigma, 0.05)
        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.clamp_min(torch.linalg.norm(epsilon_hat, dim=1, keepdim=True), 1e-10)
        delta_gamma = epsilon_hat_norm - sigma_y / (2 * mu)
        cond_yield = (delta_gamma > 0).view(-1, 1, 1)
        yield_epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        yield_F = torch.matmul(torch.matmul(U, torch.diag_embed(yield_epsilon.exp())), Vh)
        return torch.where(cond_yield, yield_F, F)


class LearnableDruckerPragerPlasticity(_LearnableMixin, P_.DruckerPragerPlasticity):
    """Upstream forward with friction_angle, cohesion, E, nu taken from the parameters."""

    def __init__(self, cfg: dict = None) -> None:
        Plasticity.__init__(self)
        self._init_params(cfg, with_dp=True)

    def forward(self, F: Tensor, log_E=None, nu=None) -> Tensor:
        E = self.log_E_sim().exp()
        nu = self.nu()
        sin_phi = torch.sin(torch.deg2rad(self.friction_angle()))
        alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        cohesion = self.cohesion()
        mu = (E / (2 * (1 + nu))).reshape(-1, 1)
        la = (E * nu / ((1 + nu) * (1 - 2 * nu))).reshape(-1, 1)
        U, sigma, Vh = self.svd(F)
        sigma = torch.clamp_min(sigma, 0.05)
        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.clamp_min(torch.linalg.norm(epsilon_hat, dim=1, keepdim=True), 1e-10)
        expand_epsilon = torch.ones_like(epsilon) * cohesion
        shifted_trace = trace - cohesion * self.dim
        cond_yield = (shifted_trace < 0).view(-1, 1)
        delta_gamma = epsilon_hat_norm + (self.dim * la + 2 * mu) / (2 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (torch.clamp_min(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat
        epsilon = torch.where(cond_yield, compress_epsilon, expand_epsilon)
        return torch.matmul(torch.matmul(U, torch.diag_embed(epsilon.exp())), Vh)


_REGISTRY = {
    "CorotatedElasticity": LearnableCorotatedElasticity,
    "StVKElasticity": LearnableStVKElasticity,
    "SigmaElasticity": LearnableSigmaElasticity,
    "VolumeElasticity": LearnableVolumeElasticity,
    "FluidElasticity": LearnableFluidElasticity,
    "IdentityPlasticity": LearnableIdentityPlasticity,
    "SigmaPlasticity": LearnableSigmaPlasticity,
    "VonMisesPlasticity": LearnableVonMisesPlasticity,
    "DruckerPragerPlasticity": LearnableDruckerPragerPlasticity,
}


def make_learnable(name: str, cfg: dict = None):
    """Instantiate the learnable variant of the upstream expert `name` (also accepts 'Learnable<name>')."""
    key = name[len("Learnable"):] if name.startswith("Learnable") else name
    if key not in _REGISTRY:
        raise KeyError(f"no learnable variant of {name}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key](cfg)
