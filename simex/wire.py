"""JSON transport for numerical observations, no pickle or executable payloads."""
import base64
import numpy as np


def encode(value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return {'__ndarray__': base64.b64encode(a.tobytes()).decode(), 'dtype': a.dtype.str, 'shape': list(a.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, dict) and '__ndarray__' in value:
        dtype = np.dtype(value['dtype'])
        if dtype.hasobject or dtype.kind not in 'biuf':
            raise ValueError('Only numerical arrays are allowed')
        return np.frombuffer(base64.b64decode(value['__ndarray__']), dtype=dtype).reshape(value['shape']).copy()
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value
