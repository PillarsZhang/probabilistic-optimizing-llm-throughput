import atexit
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import lmdb
from torch import Tensor, Any, Callable
import torch
from torch.utils.data import Dataset
import srsly
from loguru import logger

from .dist_lmdb import LMDB_MAP_SIZE
from ..utils.io import get_lmdb_keys, use_file
from ..utils.numerical import repr_for_dataclass
from ..utils.parallel import simple_cache

TSplitName = Literal["train", "valid", "test"]


@dataclass
class TrainableRecord:
    tokens: tuple
    hidden: Tensor
    prev_length: int
    post_lengths: list[int]
    tags: tuple[str, ...]
    dataset_idx: int

    def __repr__(self):
        return repr_for_dataclass(self)

    def dumps(self):
        return srsly.pickle_dumps(asdict(self))

    @classmethod
    def loads(cls, b: bytes):
        return cls(**srsly.pickle_loads(b))


TTrainableLMDBTransform = Callable[[TrainableRecord], Any]


class TrainableLMDBTransform:
    def __init__(
        self, max_num_samples: int, max_seq_len: int = None, quantile: float = None
    ) -> None:
        self.max_num_samples = max_num_samples
        self.max_seq_len = max_seq_len
        self.quantile = quantile

    def __call__(self, r: TrainableRecord):
        # Allow to limit the number of samples
        lengths = torch.tensor(r.post_lengths[: self.max_num_samples]) + r.prev_length

        src = r.hidden.float()

        if self.quantile is None:
            trg = torch.bincount(lengths, minlength=self.max_seq_len + 1).float()
            trg /= trg.sum()
        elif self.quantile == "max":
            trg = torch.tensor(lengths.float().max())[None]
        elif self.quantile == "mean":
            trg = torch.tensor(lengths.float().mean())[None]
        else:
            trg = torch.quantile(lengths.float(), self.quantile, interpolation="linear")[None]
        return src, trg

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}("
            f"max_num_samples={self.max_num_samples}, "
            f"max_seq_len={self.max_seq_len})"
        )


class TrainableLMDBDataset(Dataset):
    def __init__(self, trainable_lmdb: Path, transform: TTrainableLMDBTransform = None):
        p = use_file(trainable_lmdb, new=False).as_posix()
        self.lmdb_read_kwargs = dict(
            path=p,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=LMDB_MAP_SIZE,
        )
        env: lmdb.Environment
        with lmdb.open(**self.lmdb_read_kwargs) as env:
            with env.begin() as txn:
                lmdb_keys = get_lmdb_keys(txn)
                self.max_num_samples: int = srsly.json_loads(txn.get(b"__max_num_samples__"))
                self.generation_config_dic = srsly.json_loads(txn.get(b"__generation_config__"))
                self.max_seq_len: int = self.generation_config_dic["max_seq_len"]
                self.split_h_dic: dict[str, set] = srsly.pickle_loads(txn.get(b"__split_h__"))
                self.idx_to_h_dic: dict[int, set] = srsly.pickle_loads(txn.get(b"__idx_to_h__"))
                self.idx_to_prompt_h_dic: dict[int, set] = srsly.pickle_loads(
                    txn.get(b"__idx_to_prompt_h__")
                )

        # Strange! Perhaps there is an issue with LMDB and the file system?
        # len(lmdb_keys)
        # 1419221
        # len(self.h_set)
        # 1419239

        self.h_set = set.union(*self.idx_to_h_dic.values())
        if not self.h_set == lmdb_keys:
            logger.warning(
                f"idx_to_h conflict for self.h_set == lmdb_keys: "
                f"len(lmdb_keys)={len(lmdb_keys)}, len(lmdb_keys)={len(lmdb_keys)}"
            )
            # TODO: Remove error bypass
            # raise Exception("idx_to_h conflict")

        self.prompt_h_set = set.union(*self.idx_to_prompt_h_dic.values())
        if not self.prompt_h_set <= self.h_set:
            logger.warning(
                f"idx_to_prompt_h conflict for self.prompt_h_set <= self.h_set: "
                f"len(self.prompt_h_set - self.h_set)={len(self.prompt_h_set - self.h_set)}"
            )
            # TODO: Remove error bypass
            # raise Exception("idx_to_prompt_h conflict")

        # Hash helps with random sorting
        self.selected_h_tup = sorted(self.h_set)

        self.env: lmdb.Environment = None
        self.operations = ()
        self.transform = transform

    def select(
        self, split_name: TSplitName = None, prompt_only: bool = False, max_num_items: int = None
    ):
        obj = copy(self)
        obj.operations += (
            f"filter(prompt_only={prompt_only}, "
            f"split_name={split_name}, "
            f"max_num_items={max_num_items})",
        )
        tup = sorted(self.h_set)
        if tup != obj.selected_h_tup:
            raise Exception("You did not select from the raw data")
        if split_name is not None:
            tup = [h for h in tup if h in self.split_h_dic[split_name]]
        if prompt_only:
            tup = [h for h in tup if h in self.prompt_h_set]
        if max_num_items is not None:
            if len(tup) < max_num_items:
                raise ValueError(f"num_items({len(tup)}) < max_num_items({max_num_items})!")
            tup = tup[:max_num_items]
        obj.selected_h_tup = tup
        return obj

    def _lmdb_env_reopen(self):
        """Inspired by
        https://junyonglee.me/research/pytorch/How-to-use-LMDB-with-PyTorch-DataLoader/
        to avoid
        Caught BadRslotError in DataLoader worker process 1.
        lmdb.BadRslotError: mdb_txn_renew: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
        """
        env: lmdb.Environment = lmdb.open(**self.lmdb_read_kwargs)
        atexit.register(env.close)
        self.env = env

    def __len__(self):
        return len(self.selected_h_tup)

    def __getitem__(self, index: int | bytes):
        if not self.env:
            self._lmdb_env_reopen()

        if isinstance(index, int):
            h_bytes = self.selected_h_tup[index]
        elif isinstance(index, bytes):
            h_bytes = index
        else:
            raise ValueError(f"Unrecognized index type: {type(index)} {repr(index)}")

        with self.env.begin() as txn:
            r_bytes = txn.get(h_bytes)
        r = TrainableRecord.loads(r_bytes)

        if self.transform is None:
            return r
        else:
            return self.transform(r)

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}("
            f"length={len(self)}, "
            f"operations={'->'.join(self.operations)}, "
            f"transform={repr(self.transform)})"
        )


class TrainableLMDBBertTransform:
    def __init__(
        self,
        dataset: Dataset,
        max_num_samples: int,
        max_seq_len: int = None,
        quantile: float = None,
    ) -> None:
        self.dataset = dataset
        self.max_num_samples = max_num_samples
        self.max_seq_len = max_seq_len
        self.quantile = quantile

    def __call__(self, r: TrainableRecord):
        example = self.dataset[r.dataset_idx]
        src = self.convert_alpaca_to_prompt(example)

        lengths = torch.tensor(r.post_lengths[: self.max_num_samples]) + r.prev_length
        if self.quantile is None:
            trg = torch.bincount(lengths, minlength=self.max_seq_len + 1).float()
            trg /= trg.sum()
        elif self.quantile == "max":
            trg = torch.tensor(lengths.float().max())[None]
        elif self.quantile == "mean":
            trg = torch.tensor(lengths.float().mean())[None]
        else:
            trg = torch.quantile(lengths.float(), self.quantile, interpolation="linear")[None]
        return src, trg

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}("
            f"max_num_samples={self.max_num_samples}, "
            f"max_seq_len={self.max_seq_len})"
        )

    def convert_alpaca_to_prompt(self, example) -> str:
        if example["input"] == "":
            return example["instruction"]
        else:
            return example["instruction"] + "\n" + example["input"]
