import math
import secrets
import warnings
from pathlib import Path
import lmdb
import srsly
from srsly.util import JSONInputBin, JSONOutputBin

try:
    from isal import isal_zlib as _zlib
except ImportError:
    import zlib as _zlib

    warnings.warn(
        "Using standard zlib, which is slower than isal. Install 'isal' for better performance.",
        ImportWarning,
    )


StrPath = str | Path


def use_dir(dir_path: StrPath, new: bool = True) -> Path:
    dir_path = Path(dir_path).expanduser()
    if new:
        dir_path.mkdir(exist_ok=True, parents=True)
    else:
        assert dir_path.is_dir(), Exception(f"{dir_path} is not a directory")
    return dir_path


def use_file(file_path: StrPath, new: bool = True) -> Path:
    file_path = Path(file_path).expanduser()
    if new:
        file_path.parent.mkdir(exist_ok=True, parents=True)
    else:
        assert file_path.is_file(), Exception(f"{file_path} is not a file")
    return file_path


def srsly_msgpack_zlib_loads(data: bytes, use_list: bool = True) -> JSONOutputBin:
    """Decompress zlib bytes than deserialize msgpack bytes to a Python object"""
    return srsly.msgpack_loads(_zlib.decompress(data), use_list)


def srsly_msgpack_zlib_dumps(data: JSONInputBin) -> bytes:
    """Serialize an object to a msgpack byte string than compress zlib byte."""
    return _zlib.compress(srsly.msgpack_dumps(data))


def srsly_pickle_zlib_loads(data: bytes) -> JSONOutputBin:
    """Decompress zlib bytes than deserialize pickle bytes to a Python object"""
    return srsly.pickle_loads(_zlib.decompress(data))


def srsly_pickle_zlib_dumps(data: JSONInputBin, protocol: int = None) -> bytes:
    """Serialize an object to a pickle byte string than compress zlib byte."""
    return _zlib.compress(srsly.pickle_dumps(data, protocol))


def get_lmdb_keys(txn: lmdb.Transaction, skip_attr=True) -> set[bytes]:
    keys = set(txn.cursor().iternext(values=False))
    if skip_attr:
        keys = {x for x in keys if not x.startswith(b"__")}
    return keys


def get_lmdb_attrs(txn: lmdb.Transaction) -> set[bytes]:
    keys = set(txn.cursor().iternext(values=False))
    keys = {x for x in keys if x.startswith(b"__")}
    return keys


def get_random_hex(length: int = 8):
    """Generate a random hexadecimal string of specified length"""
    return secrets.token_hex(math.ceil(length / 2))[:length]
