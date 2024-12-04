from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import safetensors.torch
from torch import Tensor, Callable
import torch
from torch.utils.data import Dataset
import srsly
from loguru import logger
import safetensors
from itertools import groupby

from ..utils.numerical import repr_for_dataclass, split_dataset


TSplitName = Literal["train", "valid"]
SPLIT_NAMES = "train", "valid"
SPLIT_PROPS = 0.8, 0.2
SPLIT_SEED = 20240523

MAX_NEW_LEN = 512


@dataclass
class SschSampleRecord:
    # tokens: tuple
    hidden: Tensor
    # prev_length: int
    post_lengths: list[int]
    tags: tuple[str, ...]
    dataset_idx: int

    def __repr__(self):
        return repr_for_dataclass(self)


class SschSampleTransform:
    def __init__(self, max_new_len, max_num_samples: int = None) -> None:
        self.max_new_len = max_new_len
        self.max_num_samples = max_num_samples

    def __call__(self, r: SschSampleRecord):
        # Allow to limit the number of samples
        lengths = torch.tensor(r.post_lengths[: self.max_num_samples])

        src = r.hidden.float()
        trg = torch.bincount(lengths, minlength=self.max_new_len + 1).float()
        trg /= trg.sum()
        return src, trg

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}("
            f"max_num_samples={self.max_num_samples}, "
            f"max_new_len={self.max_new_len})"
        )


class SschSampleDataset(Dataset):
    def __init__(
        self,
        embedding_path: Path,
        length_path: Path,
        llm_embedding_device: str = "cuda",
        transform: Callable = None,
    ):
        llm_embedding = safetensors.torch.load_file(embedding_path, device=llm_embedding_device)
        logger.debug(f"Loaded {embedding_path=}, {len(llm_embedding)=}")

        # *.jsonl or single file
        sample_lst = []
        if length_path.is_dir():
            length_paths = sorted(length_path.glob("*.jsonl"))
            for p in length_paths:
                sample_lst.extend(srsly.read_jsonl(p))
            logger.debug(f"Recursively loaded {len(length_paths)} jsonl files from {length_path=}")
        else:
            sample_lst.extend(srsly.read_jsonl(length_path))
            logger.debug(f"Loaded {length_path=}")
        for s in sample_lst:
            s["id"] = int(s["id"])

        # Prepare llm_length
        # sample_item_example = {"id":"15275","ts":1716394766.3604156971,"temp":0.5,"seed":75,
        # "input":"...","L_gt":109,"L_t0.5":291,"L_input":196,"output_t0.5":"..."}

        llm_temp = sample_lst[0]["temp"]
        # llm_prev_length_key = f"L_input"
        llm_post_length_key = f"L_t{llm_temp}"

        self.max_new_len = MAX_NEW_LEN

        # group sample_item by id
        keyfunc = lambda x: x["id"]
        sample_lst = sorted(sample_lst, key=keyfunc)
        self.record_lst = []
        for k, g in groupby(sample_lst, key=keyfunc):
            g = sorted(g, key=lambda x: x["seed"])

            # Assert all prev_length, input are the same
            assert all(x["input"] == g[0]["input"] for x in g)

            # assert all(x[llm_prev_length_key] == g[0][llm_prev_length_key] for x in g)
            post_lengths = [x[llm_post_length_key] for x in g]
            post_lengths = [x if x <= self.max_new_len else self.max_new_len for x in post_lengths]

            r = SschSampleRecord(
                hidden=llm_embedding[str(k)],
                post_lengths=post_lengths,
                tags=tuple(),
                dataset_idx=k,
            )
            self.record_lst.append(r)
        logger.debug(f"Prepared {len(self.record_lst)=}")

        num_samples_lst = [len(r.post_lengths) for r in self.record_lst]
        self.max_num_samples = max(num_samples_lst)
        if not all(x == self.max_num_samples for x in num_samples_lst):
            logger.warning(f"{self.max_num_samples=} is not consistent!")

        # Split set
        self.raw_num_records = len(self.record_lst)
        split_ind_tups = split_dataset(range(self.raw_num_records), SPLIT_PROPS, SPLIT_SEED)
        self.split_ind_dic = dict(zip(SPLIT_NAMES, split_ind_tups))
        for split_name, split_inds in self.split_ind_dic.items():
            logger.debug(f"{split_name=}, {len(split_inds)=}")
            for ind in split_inds:
                self.record_lst[ind].tags += (split_name,)

        self.operations = ()
        self.transform = transform

    def select(self, split_name: TSplitName = None, max_num_items: int = None):
        obj = copy(self)
        obj.operations += (f"filter(split_name={split_name}, max_num_items={max_num_items})",)
        lst = self.record_lst
        if len(lst) != self.raw_num_records:
            raise Exception("You did not select from the raw data")
        if split_name is not None:
            lst = [lst[ind] for ind in self.split_ind_dic[split_name]]
        if max_num_items is not None:
            if len(lst) < max_num_items:
                raise ValueError(f"num_items({len(lst)}) < max_num_items({max_num_items})!")
            lst = lst[:max_num_items]
        obj.record_lst = lst
        return obj

    def __len__(self):
        return len(self.record_lst)

    def __getitem__(self, ind: int):
        r = self.record_lst[ind]

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


class SschSampleBertTransform:
    def __init__(self, conv_lst: list[dict], max_new_len, max_num_samples: int = None) -> None:
        self.id_to_text = {int(c["id"]): c["conversations"][0]["value"] for c in conv_lst}
        self.max_new_len = max_new_len
        self.max_num_samples = max_num_samples

    def __call__(self, r: SschSampleRecord):
        src = self.id_to_text[r.dataset_idx]
        lengths = torch.tensor(r.post_lengths[: self.max_num_samples])
        trg = torch.bincount(lengths, minlength=self.max_new_len + 1).float()
        trg /= trg.sum()
        return src, trg

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}("
            f"max_num_samples={self.max_num_samples}, "
            f"max_new_len={self.max_new_len})"
        )
