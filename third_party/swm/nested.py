"""Dotted lookup used by the vendored variation spaces."""
def get_in(obj, path):
    for key in (path.split('.') if isinstance(path, str) else path):
        obj = obj[key]
    return obj
