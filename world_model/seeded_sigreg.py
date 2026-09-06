"""Pinned SIGReg with Claude's lazy, checkpointed private sketch stream.

Numerics/buffers reused from MIT LeWM module.py at commit
8edfeb336732b5f3ce7b8b210d0ba370a09e2cac, https://github.com/lucas-maes/le-wm.
Claude MCP contribution: runs/projection_training_2026-09-06/claude.
Integration preserves the source's doubled quadrature weights, float32 draws
and autocast matmul reduction; these differ from the generic draft.
"""
import torch
from third_party.lewm.module import SIGReg


class SeededSIGReg(SIGReg):
    """An opt-in stream; legacy SIGReg remains untouched.

    Active RNG states are device-specific. Resume on the same device; changing
    projection count, seed or knots requires a fresh experiment. Fixed unit
    directions are accepted only for numerical checks and never advance RNG.
    """
    def __init__(self, knots=17, num_proj=1024, seed=0):
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (knots, num_proj, seed)):
            raise ValueError('knots, num_proj and seed must be integers')
        if knots < 2 or num_proj < 1:
            raise ValueError('At least two knots and one projection are required')
        super().__init__(knots, num_proj)
        self.knots, self.seed = knots, seed
        self._generator = None
        self._pending_state = None

    def _ensure_generator(self, device):
        device = torch.device(device)
        if self._generator is not None:
            if self._generator.device != device:
                raise RuntimeError('Active sketch stream cannot change device')
            return self._generator
        if self._pending_state is not None and self._pending_state[1] != str(device):
            raise RuntimeError('Checkpointed sketch stream cannot change device')
        generator = torch.Generator(device=device)
        if self._pending_state is None:
            generator.manual_seed(self.seed)
        else:
            generator.set_state(self._pending_state[0])
        self._generator, self._pending_state = generator, None
        return generator

    def get_extra_state(self):
        if self._generator is not None:
            raw, device = self._generator.get_state().clone(), str(self._generator.device)
        elif self._pending_state is not None:
            raw, device = self._pending_state[0].clone(), self._pending_state[1]
        else:
            raw, device = None, None
        return dict(version=1, knots=self.knots, num_proj=self.num_proj, seed=self.seed,
                    generator_state=raw, generator_device=device)

    def set_extra_state(self, state):
        required = {'version', 'knots', 'num_proj', 'seed', 'generator_state', 'generator_device'}
        if not isinstance(state, dict) or not required.issubset(state):
            raise RuntimeError('Incomplete private sketch checkpoint')
        expected = dict(version=1, knots=self.knots, num_proj=self.num_proj, seed=self.seed)
        if any(state[key] != value for key, value in expected.items()):
            raise RuntimeError('Incompatible private sketch configuration')
        raw, device = state['generator_state'], state['generator_device']
        if (raw is None) != (device is None):
            raise RuntimeError('Incomplete private sketch stream')
        if raw is not None:
            if not isinstance(raw, torch.Tensor) or raw.dtype != torch.uint8 or raw.ndim != 1:
                raise RuntimeError('Private sketch RNG state must be a uint8 vector')
            device = str(torch.device(device))
        self._generator = None
        self._pending_state = None if raw is None else (raw.detach().cpu().clone(), device)

    def forward(self, proj, directions=None):
        if proj.ndim != 3 or not proj.is_floating_point() or min(proj.shape) < 1:
            raise ValueError('Expected a nonempty floating [T, B, D] tensor')
        if directions is None:
            generator = self._ensure_generator(proj.device)
            # Same draw dtype and normalization as the pinned source, independent RNG.
            directions = torch.randn(proj.size(-1), self.num_proj, device=proj.device,
                                     generator=generator)
            directions = directions.div_(directions.norm(p=2, dim=0))
        elif (directions.shape != (proj.size(-1), self.num_proj)
              or directions.device != proj.device or not directions.is_floating_point()):
            raise ValueError('Fixed directions must be floating [D, M] on the input device')
        x_t = (proj @ directions).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
