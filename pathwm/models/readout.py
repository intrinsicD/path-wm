"""Optional latent output processing; does not mutate belief or source memory."""

import torch
from torch import nn

from .modalities import Attend, ImageDecoder, position


class RecurrentOutputAdapter(nn.Module):
    """Refine local tokens against fixed context with one shared pair of blocks.

    Initially identical to the supplied context at valid positions. This permits
    matched decoder initialization. A zero residual gain initially blocks gradients
    to the refinement blocks; the gain learns first. Iterations add computation,
    not registered parameters. Read once per emitted sequence, not per text byte.
    """

    def __init__(self, width, *, iterations=1):
        super().__init__()
        self.width = width
        self.iterations = iterations
        self.local = Attend(width)
        self.cross = Attend(width)
        self.gain = nn.Parameter(torch.zeros(()))

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        if type(value) is not int or value < 0:
            raise ValueError("Readout iterations must be a nonnegative integer")
        self._iterations = value

    def extra_repr(self):
        return f"width={self.width}, iterations={self.iterations}"

    def forward(self, context, *, valid=None, trace=None):
        if context.ndim != 3 or context.shape[-1] != self.width:
            raise ValueError("Readout context must be [B,N,width]")
        if valid is None:
            valid = torch.ones(
                context.shape[:2], device=context.device, dtype=torch.bool
            )
        if (
            valid.shape != context.shape[:2]
            or valid.dtype != torch.bool
            or valid.device != context.device
            or not valid.any(1).all()
        ):
            raise ValueError(
                "Readout needs boolean validity and a valid token per sample"
            )
        source = context.masked_fill(~valid[..., None], 0)
        if not torch.isfinite(source).all():
            raise ValueError("Valid context must be finite")
        local = source
        for i in range(self.iterations):
            local = self.local(
                local, local, valid=valid, trace=trace, name=f"readout.loop.{i}.local"
            )
            local = self.cross(
                local, source, valid=valid, trace=trace, name=f"readout.loop.{i}.cross"
            )
            local = local.masked_fill(~valid[..., None], 0)
            if trace is not None:
                trace[f"readout.loop.{i}.tokens"] = local.detach().cpu().clone()
        return source + self.gain.tanh() * (local - source)


class TemporalImageDecoder(nn.Module):
    """Requested-time conditioning around a shared native image decoder.

    Does not roll out world dynamics or receive target frames. Temporal conditioning
    alone is not a trained future-video predictor. Times are explicit relative
    output coordinates, not fresh observation events.
    """

    def __init__(
        self, width, image_size=16, *, time_conditioning="context", palette_size=0
    ):
        super().__init__()
        self.width = width
        self.palette_size = palette_size
        self.time_conditioning = time_conditioning
        self.time_projection = nn.Linear(width, width)
        self.image = ImageDecoder(width, image_size, palette_size=palette_size)

    @property
    def time_conditioning(self):
        return self._time_conditioning

    @time_conditioning.setter
    def time_conditioning(self, value):
        if value not in ("context", "query"):
            raise ValueError("Unknown video time conditioning")
        if self.palette_size and value != "query":
            raise ValueError("Video palette requires query-side time")
        self._time_conditioning = value

    def forward(self, context, times, *, valid=None, trace=None):
        if (
            times.ndim != 1
            or times.numel() < 1
            or not times.is_floating_point()
            or times.device != context.device
            or not torch.isfinite(times).all()
            or not (times[1:] > times[:-1]).all()
        ):
            raise ValueError(
                "Video output times must be finite and strictly increasing"
            )
        b, n, width = context.shape
        code = self.time_projection(position(times.to(context.dtype), self.width))
        if self.time_conditioning == "context":
            # Preserve the original numerical path and checkpoint parameters.
            values = (context[:, None] + code[None, :, None]).reshape(
                b * len(times), n, width
            )
            offset = None
        else:
            values = (
                context[:, None]
                .expand(-1, len(times), -1, -1)
                .reshape(b * len(times), n, width)
            )
            offset = (
                code[None, :, None]
                .expand(b, -1, -1, -1)
                .reshape(b * len(times), 1, width)
            )
        mask = (
            None
            if valid is None
            else valid[:, None].expand(-1, len(times), -1).reshape(b * len(times), n)
        )
        decoded = self.image(values, trace=trace, valid=mask, query_offset=offset)
        return decoded.reshape(b, len(times), *decoded.shape[1:])
