"""Smaller display encoding with exact pixel preservation; source PNGs stay intact."""
import base64
import io
from pathlib import Path
from PIL import Image


def image_data_uri(path):
    original=Path(path).read_bytes()
    with Image.open(io.BytesIO(original)) as source:
        # Keep profile-bearing or animated inputs in their original representation.
        if source.format!='PNG' or any(key in source.info for key in ('icc_profile','gamma','chromaticity')) or getattr(source,'n_frames',1)!=1:
            return 'data:'+Image.MIME.get(source.format,'image/png')+';base64,'+base64.b64encode(original).decode()
        rgba=source.convert('RGBA'); buffer=io.BytesIO()
        rgba.save(buffer,format='WEBP',lossless=True,exact=True,method=4)
        encoded=buffer.getvalue()
        with Image.open(io.BytesIO(encoded)) as decoded:
            if decoded.size!=rgba.size or decoded.convert('RGBA').tobytes()!=rgba.tobytes():
                raise ValueError('Scientific preview encoding changed RGBA pixels')
    mime,payload=('image/webp',encoded) if len(encoded)<len(original) else ('image/png',original)
    return 'data:'+mime+';base64,'+base64.b64encode(payload).decode()
