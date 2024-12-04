import hashlib
import json
from numbers import Number
import os
from pathlib import Path
import time
from typing import Iterable, Any


def get_script_id(script_path: str) -> Path:
    script_path = Path(script_path).absolute()
    base_path = Path("scripts").absolute()
    try:
        relative_path = script_path.relative_to(base_path)
        script_id = relative_path.with_suffix("").as_posix()
    except ValueError:
        script_id = script_path.stem
    return script_id


def pick_keys(dic: dict, keys: Iterable):
    return {k: dic[k] for k in keys}


def format_dict(
    dic: dict[str, Any],
    ndigits: int = 2,
    prefix: str = "",
    sep: str = ", ",
    colon: str = ": ",
) -> str:
    new_dic: dict[str, str] = dict()
    for k, v in dic.items():
        new_dic[str(k)] = str(round(v, ndigits) if isinstance(v, Number) else v)
    return prefix + sep.join(f"{k}{colon}{v}" for k, v in new_dic.items() if not k.startswith("_"))


def hash_obj(obj: Any, bit_length: int = 32):
    obj_bytes = json.dumps(obj).encode()
    hash_bytes = hashlib.sha256(obj_bytes).digest()
    hash_int = int.from_bytes(hash_bytes, byteorder="big") & ((1 << bit_length) - 1)
    return hash_int


def format_bytes(num: int, ndigits: int = 3) -> str:
    """Format bytes to MB, GB, etc."""
    for unit in ("Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"):
        if num < 1024:
            break
        else:
            num /= 1024
    return f"{round(num, ndigits)} {unit}"


class tic:
    """Simple reusable timer"""

    def __init__(self) -> None:
        self.t = time.time()

    def __call__(self) -> float:
        _t = self.t
        self.t = time.time()
        return self.t - _t


def get_env():
    return os.environ.get("LSDIST_ENV", "production")


def get_ver():
    return os.environ.get("LSDIST_VER", "")
