"""Self-contained native image codes; shared hierarchy/decoder, no RGB lookup.

Only the actual RGB64/width64/two-scale native configuration is supported. The
serialized payload is the processed fine FeatureScale. Coarse and slots are
computed, never stored as independent truth. This interface establishes replay,
not pixel fidelity. It does not invoke or require a generator.
"""
import math
from pathlib import Path
import torch
from pathwm.io import atomic_torch, state_hash
from .multiscale import FeatureScale

SCHEMA = 'native-image-code-v1'
FIELDS = ('values', 'times', 'valid', 'ends', 'content_times')


def _native(perception):
    e = perception.encoder
    h = e.pyramid
    if (perception.width != 64 or perception.slot_count != 7 or
        perception.slot_attention.iterations != 3 or e.stem.patch_size != 4 or
        e.video or e.code_width != 16 or len(h.stages) != 2 or len(h.merges) != 1 or
        h.fusion or h.layer_readout is not None or
        any(len(s.blocks) != 1 or s.window is not None for s in h.stages) or
        h.merges[0].factors != (1, 2, 2) or h.merges[0].packed or
        h.merges[0].cross_attention is None):
        raise ValueError('Image code requires the native full configuration')
    if any(m.training for m in perception.modules()):
        raise ValueError('Image code replay requires evaluation mode')
    if next(perception.encoder.parameters()).dtype != torch.float32:
        raise ValueError('Native image code replay requires float32')


def _validate(code):
    required = {'schema','encoding','encoder_sha256','fine','condition','condition_time','provenance'}
    if not isinstance(code, dict) or set(code) != required or code['schema'] != SCHEMA:
        raise ValueError('Invalid image code schema')
    if code['encoding'] != 'RGB64-fp32-frame-local-zero' or code['condition_time'] is not None:
        raise ValueError('Unsupported image encoding convention')
    if not isinstance(code['encoder_sha256'], str) or len(code['encoder_sha256']) != 64:
        raise ValueError('Invalid image code weights binding')
    f = code['fine']
    if not isinstance(f, dict) or set(f) != set(FIELDS) | {'grid'} or tuple(f['grid']) != (1,16,16):
        raise ValueError('Invalid native fine grid')
    x = f['values']
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (256,64)
        or len(x) < 1 or x.dtype != torch.float32 or not torch.isfinite(x).all()):
        raise ValueError('Invalid native fine values')
    b = len(x)
    for k in ('times','content_times','ends','valid'):
        v=f[k]
        dtype = torch.bool if k=='valid' else torch.int64 if k=='ends' else torch.float32
        if not isinstance(v, torch.Tensor) or v.shape != (b,256) or v.dtype != dtype:
            raise ValueError('Invalid native fine metadata')
        if not bool(v.all() if k=='valid' else (v==0).all()):
            raise ValueError('Native codes require complete frame-local zero metadata')
    c=code['condition']
    if not isinstance(c,torch.Tensor) or c.shape!=(b,16) or c.dtype!=torch.float32 or not (c==0).all():
        raise ValueError('Native code condition must be explicit zero [B,16]')
    provenance=code['provenance']
    if not isinstance(provenance,list) or len(provenance)!=b:
        raise ValueError('One provenance record per image required')
    for p in provenance:
        if not isinstance(p,dict) or set(p)!={'kind','id','available_at'}:
            raise ValueError('Invalid image provenance fields')
        if p['kind'] not in ('observed','generated') or not isinstance(p['id'],str) or not p['id']:
            raise ValueError('Invalid image provenance kind or identifier')
        if type(p['available_at']) not in (int,float) or not math.isfinite(p['available_at']) or p['available_at']<0:
            raise ValueError('Invalid image provenance time')


def make_image_code(perception, pyramid, *, provenance):
    """Detach a native observed/generated fine export into a portable CPU payload.

    Provenance identifies the caller's source/sample but is never dereferenced.
    This is persistence, not a differentiable producer-training operation.
    """
    _native(perception)
    if len(pyramid.scales)!=2 or pyramid.condition_time is not None:
        raise ValueError('Expected native frame-local pyramid')
    fine=pyramid.scales[0]
    code=dict(schema=SCHEMA, encoding='RGB64-fp32-frame-local-zero',
              encoder_sha256=state_hash(perception.encoder),
              fine={**{k:getattr(fine,k).detach().cpu().clone() for k in FIELDS},'grid':fine.grid},
              condition=torch.zeros(len(fine.values),16), condition_time=None,
              provenance=[dict(p) for p in provenance])
    _validate(code)
    return code


def code_from_values(perception, values, *, provenance):
    """Wrap native fine token values [B,256,64] (e.g. a generated edit) with the
    frame-local zero metadata of the native encoder. Values carry no image, source
    event or blob; provenance is caller-declared and never dereferenced."""
    _native(perception)
    if (not isinstance(values, torch.Tensor) or values.ndim != 3 or values.shape[1:] != (256, 64)
            or values.dtype != torch.float32):
        raise ValueError('Native fine values must be [B,256,64]')
    b = len(values)
    zeros = torch.zeros(b, 256)
    code = dict(schema=SCHEMA, encoding='RGB64-fp32-frame-local-zero',
                encoder_sha256=state_hash(perception.encoder),
                fine=dict(values=values.detach().cpu().clone(), times=zeros.clone(),
                          valid=torch.ones(b, 256, dtype=torch.bool), ends=torch.zeros(b, 256, dtype=torch.long),
                          content_times=zeros.clone(), grid=(1, 16, 16)),
                condition=torch.zeros(b, 16), condition_time=None,
                provenance=[dict(p) for p in provenance])
    _validate(code)
    return code


def save_image_code(path, code):
    _validate(code)
    atomic_torch(Path(path),code)


def load_image_code(path):
    code=torch.load(path,map_location='cpu',weights_only=True)
    _validate(code)
    return code


def code_pyramid(perception, code):
    """Recreate the native consumer view, using no image, source store or sampler."""
    _native(perception)
    _validate(code)
    if state_hash(perception.encoder)!=code['encoder_sha256']:
        raise ValueError('Image code weights differ from the bound model')
    device=next(perception.parameters()).device
    f=code['fine']
    fine=FeatureScale(**{k:f[k].to(device) for k in FIELDS},grid=tuple(f['grid']))
    return perception.encoder.pyramid.continue_from_fine(fine,code['condition'].to(device),condition_time=None)


def decode_image_code(perception, code):
    """Return the normal native Percept; exact code replay does not imply fidelity."""
    if perception.decoder.hidden != 32:
        raise ValueError('Native decode requires decoder width32')
    return perception.from_pyramid(code_pyramid(perception,code))
