from typing import Any, Callable
import lmdb
import atexit
import srsly
from torch import Tensor
import torch
from torch.utils.data import Dataset
from pathlib import Path

from ..sample.llama import LlamaSampleResult, LlamaSampleKey

LMDB_MAP_SIZE = 2 << 40

TDistLMDBTransform = Callable[[LlamaSampleKey, LlamaSampleResult], Any]


class DistLMDBTransform:
    def __init__(self) -> None:
        pass

    def __call__(self, key: LlamaSampleKey, result: LlamaSampleResult):
        src = torch.tensor(result.prompt_hiddens[-1, -1, :])
        trg = 1 / (1 + torch.tensor(result.generation_length_lst) / result.prompt_length)
        return src, trg


class DistLMDBDataset(Dataset):
    def __init__(
        self, *dist_lmdbs: tuple[Path], sort: bool = True, transform: TDistLMDBTransform = None
    ):
        """Initializes the dataset object with multiple LMDB files.

        No duplicate key in multiple databases.
        """

        # Just merge and creating a reverse lookup path
        lookup_key_bytes_to_key: dict[bytes, LlamaSampleKey] = dict()
        lookup_key_bytes_to_env_path: dict[bytes, str] = dict()

        key_bytes_lst = []
        env_path_lst = []

        for dist_lmdb in dist_lmdbs:
            env: lmdb.Environment = lmdb.open(
                dist_lmdb.as_posix(), subdir=False, readonly=True, map_size=LMDB_MAP_SIZE
            )
            env_path = env.path()
            env_path_lst.append(env_path)

            with env.begin() as txn:
                key_bytes_tup = tuple(
                    x for x in txn.cursor().iternext(values=False) if not x.startswith(b"__")
                )
            env.close()

            for key_bytes in key_bytes_tup:
                if key_bytes not in lookup_key_bytes_to_env_path:
                    lookup_key_bytes_to_env_path[key_bytes] = env_path
                else:
                    raise ValueError(
                        f"{key_bytes} already in lookup dict from "
                        f"{lookup_key_bytes_to_env_path[key_bytes]}"
                    )
                key = LlamaSampleKey.loads(key_bytes)
                key_bytes_lst.append(key_bytes)
                lookup_key_bytes_to_key[key_bytes] = key
                assert key.dumps() == key_bytes

        _lst = srsly.json_loads(b"[" + b",".join(key_bytes_lst) + b"]")
        _lst = [tuple(x) for x in key_bytes_lst]
        _lst = list(zip(_lst, key_bytes_lst))
        if sort:
            _lst.sort(key=lambda x: x[0])

        self.env_path_lst = env_path_lst
        self.key_bytes_tup = tuple(x[1] for x in _lst)
        self.lookup_env_path_to_env = dict()
        self.lookup_key_bytes_to_key = lookup_key_bytes_to_key
        self.lookup_key_bytes_to_env_path = lookup_key_bytes_to_env_path
        self.sort = sort
        self.transform = transform

    def _lmdb_env_reopen(self):
        """Inspired by
        https://junyonglee.me/research/pytorch/How-to-use-LMDB-with-PyTorch-DataLoader/
        to avoid
        Caught BadRslotError in DataLoader worker process 1.
        lmdb.BadRslotError: mdb_txn_renew: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
        """
        for env_path in self.env_path_lst:
            env: lmdb.Environment = lmdb.open(
                env_path, subdir=False, readonly=True, map_size=LMDB_MAP_SIZE
            )
            atexit.register(env.close)
            self.lookup_env_path_to_env[env_path] = env

    def __len__(self):
        return len(self.key_bytes_tup)

    def __getitem__(self, index: int | bytes | LlamaSampleKey):
        """Retrieves a data sample from the dataset given an index.

        The method supports indexing by integer, raw bytes, or a LlamaSampleKey object.
        """

        if not self.lookup_env_path_to_env:
            self._lmdb_env_reopen()

        if isinstance(index, int):
            key_bytes = self.key_bytes_tup[index]
        elif isinstance(index, bytes):
            key_bytes = index
        elif isinstance(index, LlamaSampleKey):
            key_bytes = index.dumps()
        else:
            raise ValueError(f"Unrecognized index type: {type(index)} {repr(index)}")

        if not key_bytes in self.key_bytes_tup:
            raise ValueError(f"Index does not exist: {repr(index)} {repr(key_bytes)}")

        key = self.lookup_key_bytes_to_key[key_bytes]
        env_path = self.lookup_key_bytes_to_env_path[key_bytes]
        env = self.lookup_env_path_to_env[env_path]

        with env.begin() as txn:
            result_bytes = txn.get(key_bytes)
        result = LlamaSampleResult.loads(result_bytes)

        if self.transform is None:
            return key, result
        else:
            return self.transform(key, result)

    def __repr__(self):
        """
        Provides a string representation of the DistLMDBDataset object, detailing the number of
        LMDB environments, their paths, the dataset's length, sorting status, and the transform function.
        """
        dist_lmdbs = repr(self.env_path_lst)
        return (
            f"{self.__class__.__qualname__}(length={len(self)}, "
            f"dist_lmdbs={dist_lmdbs}, "
            f"sort={self.sort}, "
            f"transform={repr(self.transform)})"
        )
