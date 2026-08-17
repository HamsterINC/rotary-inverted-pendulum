"""
Minimal torch .pth loader that doesn't require torch.
Parses the zip-based torch.save format directly using pickle + numpy.
"""
import zipfile
import pickle
import io
import numpy as np
import struct

# map torch storage type names -> numpy dtype
STORAGE_DTYPE_MAP = {
    'FloatStorage': np.float32,
    'DoubleStorage': np.float64,
    'HalfStorage': np.float16,
    'LongStorage': np.int64,
    'IntStorage': np.int32,
    'ShortStorage': np.int16,
    'CharStorage': np.int8,
    'ByteStorage': np.uint8,
    'BoolStorage': np.bool_,
    'BFloat16Storage': np.float32,  # will handle separately if needed
}

class FakeStorage:
    """Placeholder representing an untyped/typed storage backed by raw bytes."""
    def __init__(self, dtype, numel, raw_bytes):
        self.dtype = dtype
        self.numel = numel
        self.raw_bytes = raw_bytes
        self._array = np.frombuffer(raw_bytes, dtype=dtype, count=numel)

    def as_array(self):
        return self._array


class TorchUnpickler(pickle.Unpickler):
    def __init__(self, file, zipf, prefix):
        super().__init__(file)
        self.zipf = zipf
        self.prefix = prefix  # e.g. 'archive'

    def persistent_load(self, saved_id):
        # saved_id is typically ('storage', storage_type_obj_or_str, key, location, numel)
        assert saved_id[0] == 'storage'
        _, storage_type, key, location, numel = saved_id
        # storage_type may be a class-like object with __name__, or already a string
        type_name = getattr(storage_type, '__name__', None) or str(storage_type)
        dtype = STORAGE_DTYPE_MAP.get(type_name, np.float32)
        data_path = f"{self.prefix}/data/{key}"
        raw = self.zipf.read(data_path)
        return FakeStorage(dtype, numel, raw)

    def find_class(self, module, name):
        # Intercept torch-specific classes we need to fake
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return rebuild_tensor_v2
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return rebuild_parameter
        if module.startswith('torch') and 'Storage' in name:
            # Return a dummy class carrying the type name for persistent_load's storage_type
            return type(name, (), {'__name__': name})
        if module == 'collections' and name == 'OrderedDict':
            return __import__('collections').OrderedDict
        if module == 'torch' and name == 'Tensor':
            return DummyTensor
        if module == '__builtin__' or module == 'builtins':
            return getattr(__import__('builtins'), name)
        # Fallback: try to import for real (numpy, collections, etc.)
        try:
            return super().find_class(module, name)
        except Exception:
            # Give a generic placeholder object for anything unrecognized (e.g. optimizer classes)
            return type(name, (), {})


class DummyTensor:
    pass


def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    arr = storage.as_array()
    arr = arr[storage_offset:]
    # Build array using numpy strides (torch strides are in elements, numpy needs bytes)
    itemsize = arr.dtype.itemsize
    np_strides = tuple(s * itemsize for s in stride)
    if len(size) == 0:
        return arr[0:1].reshape(())
    total_needed = storage_offset + (max((s-1)*st for s, st in zip(size, stride)) + 1 if size else 1)
    out = np.lib.stride_tricks.as_strided(arr, shape=size, strides=np_strides)
    return out.copy()  # copy to make it a normal contiguous-safe array


def rebuild_parameter(data, requires_grad, backward_hooks):
    return data


def load_pth(path_or_bytes, prefix='archive'):
    """Load a torch.save .pth file. Accepts a filesystem path or raw bytes
    (e.g. bytes read out of an outer zip, so you don't need to extract to
    disk first)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        z = zipfile.ZipFile(io.BytesIO(path_or_bytes))
    else:
        z = zipfile.ZipFile(path_or_bytes)
    names = z.namelist()
    # find data.pkl regardless of prefix folder name
    pkl_names = [n for n in names if n.endswith('data.pkl')]
    assert pkl_names, f"No data.pkl found in {path_or_bytes}"
    pkl_path = pkl_names[0]
    real_prefix = pkl_path.rsplit('/', 1)[0]
    with z.open(pkl_path) as f:
        data = f.read()
    unpickler = TorchUnpickler(io.BytesIO(data), z, real_prefix)
    obj = unpickler.load()
    return obj


def load_pth_from_sb3_zip(sb3_zip_path, inner_filename='policy.pth'):
    """Load a .pth tensor file straight out of an SB3 model save (.zip),
    without extracting anything to disk first.

    SB3's model.save() produces an OUTER zip containing (among other things)
    'policy.pth', which is itself a torch.save file -- and torch.save's
    format is ALSO a zip internally. So an SB3 checkpoint is effectively a
    zip-of-a-zip. This helper unwraps both layers in memory.
    """
    with zipfile.ZipFile(sb3_zip_path) as outer:
        inner_bytes = outer.read(inner_filename)
    return load_pth(inner_bytes)

if __name__ == '__main__':
    import sys
    obj = load_pth(sys.argv[1])
    print(type(obj))
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(k, type(v), getattr(v, 'shape', None), getattr(v, 'dtype', None))