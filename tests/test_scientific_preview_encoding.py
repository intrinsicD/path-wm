import base64
import io
import numpy as np
from PIL import Image
from viewer.scientific_preview import image_data_uri


def test_lossless_preview_preserves_every_rgba_value_including_transparent_rgb(tmp_path):
    y,x=np.mgrid[:96,:96]
    pixels=np.stack((x%256,y%256,(x+y)%256,np.where(x%3,255,0)),axis=-1).astype('uint8')
    path=tmp_path/'plot.png'; Image.fromarray(pixels).save(path,compress_level=0)
    uri=image_data_uri(path)
    assert uri.startswith('data:image/webp;base64,')
    encoded=base64.b64decode(uri.split(',',1)[1])
    assert len(encoded)<path.stat().st_size
    decoded=np.asarray(Image.open(io.BytesIO(encoded)).convert('RGBA'))
    assert np.array_equal(decoded,pixels)
