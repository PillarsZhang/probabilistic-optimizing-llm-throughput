from dataclasses import asdict, astuple, dataclass, field
from typing import TypedDict
import numpy as np
import numpy.typing as npt
import torch
import srsly
from itertools import islice
from tqdm import tqdm
from loguru import logger

from ..llama import Llama, Dialog
from ..llama.generation import B_INST, E_INST, B_SYS, E_SYS

from ..utils import tic, hash_obj
from ..utils.io import srsly_msgpack_zlib_dumps, srsly_msgpack_zlib_loads
from ..utils.numerical import repr_for_dataclass
from ..utils.data import AlpacaExample, Dialog
from ..utils.distributed import on_rank


class LlamaGenerationConfig(TypedDict):
    temperature: float
    top_p: float
    max_seq_len: int
    max_gen_len: int | None


@dataclass
class LlamaSampleResult:
    hidden_indices: tuple
    prompt_str: str
    prompt_tokens: list[int]
    prompt_length: int
    prompt_hiddens: npt.NDArray[np.float32]
    generation_tokens_lst: list[list[int]] = field(default_factory=list)
    generation_str_lst: list[str] = field(default_factory=list)
    generation_length_lst: list[int] = field(default_factory=list)
    generation_hiddens_lst: list[npt.NDArray[np.float32]] = field(default_factory=list)
    time_cost: float = field(default=None)

    def __repr__(self):
        return repr_for_dataclass(self)

    def dumps(self):
        return srsly_msgpack_zlib_dumps(asdict(self))

    @classmethod
    def loads(cls, b: bytes):
        return cls(**srsly_msgpack_zlib_loads(b))

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        k_lst = ("hidden_indices", "prompt_str", "prompt_tokens", "prompt_length")
        conds = []
        for k in k_lst:
            conds.append(cond := (v_self := getattr(self, k)) == (v_other := getattr(other, k)))
            if not cond:
                logger.error(f"Cond: self.{k}({v_self}) != other.{k}({v_other})")

        if not all(conds):
            raise ValueError("Cannot merge objects with different prompt properties")

        return self.__class__(
            hidden_indices=self.hidden_indices,
            prompt_str=self.prompt_str,
            prompt_tokens=self.prompt_tokens,
            prompt_length=self.prompt_length,
            prompt_hiddens=self.prompt_hiddens,
            generation_tokens_lst=self.generation_tokens_lst + other.generation_tokens_lst,
            generation_str_lst=self.generation_str_lst + other.generation_str_lst,
            generation_length_lst=self.generation_length_lst + other.generation_length_lst,
            generation_hiddens_lst=self.generation_hiddens_lst + other.generation_hiddens_lst,
            time_cost=(
                None
                if None in (self.time_cost, other.time_cost)
                else self.time_cost + other.time_cost
            ),
        )

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @classmethod
    def sum(cls, result_lst):
        if not result_lst:
            raise ValueError("The list is empty")

        initial = cls(
            hidden_indices=result_lst[0].hidden_indices,
            prompt_str=result_lst[0].prompt_str,
            prompt_tokens=result_lst[0].prompt_tokens,
            prompt_length=result_lst[0].prompt_length,
            prompt_hiddens=result_lst[0].prompt_hiddens,
            time_cost=None if result_lst[0] is None else 0,
        )
        return sum(result_lst, initial)


@dataclass
class LlamaSampleKey:
    shuffle_seed: int
    dataset_idx: int
    batch_size: int
    supp_round: int
    num_samples: int

    def __repr__(self):
        return repr_for_dataclass(self)

    def dumps(self):
        return srsly.json_dumps(astuple(self)).encode()

    @classmethod
    def loads(cls, b: bytes):
        return LlamaSampleKey(*srsly.json_loads(b))

    @classmethod
    def batch_loads(cls, *bs: tuple[bytes], sort: bool = False):
        _tup_lst = [tuple(x) for x in srsly.json_loads(b"[" + b",".join(bs) + b"]")]
        if sort:
            _tup_lst.sort()
        g = (LlamaSampleKey(*args) for args in _tup_lst)
        return tuple(g)


class LlamaDistSampler:
    def __init__(self, generator: Llama) -> None:
        self.generator = generator
        self.model = generator.model
        self.tokenizer = generator.tokenizer

    def sample_from_prompt(
        self,
        example: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: int = None,
        supp_round: int = 0,
        num_samples: int = 50,
        hidden_indices: tuple = (),
        prompt_only: bool = False,
    ) -> LlamaSampleResult:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        batch_size = self.model.params.max_batch_size

        dialog: Dialog = [
            {
                "role": "user",
                "content": example,
            }
        ]
        dialog_tokens: list[int] = self.convert_dialog_to_tokens(dialog)

        # How about having the same prompt share the same top-p random state?
        # But we need to accept multiple rounds of data supplementation
        dialog_seed = hash_obj((dialog_tokens, supp_round))
        dialog_trng = torch.Generator(device="cuda").manual_seed(dialog_seed)

        result = LlamaSampleResult(
            hidden_indices=hidden_indices,
            prompt_str=self.tokenizer.decode(dialog_tokens),
            prompt_tokens=dialog_tokens,
            prompt_length=len(dialog_tokens),
            prompt_hiddens=None,
        )

        def g():
            while True:
                _ret = self.generator.generate(
                    prompt_tokens=[dialog_tokens] * batch_size,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                    trng=dialog_trng,
                    hidden_indices=hidden_indices,
                )
                yield from zip(*_ret)

        toc = tic()
        pbar = tqdm(
            islice(g(), num_samples),
            total=num_samples,
            desc="Sampling",
            disable=not on_rank(),
            leave=False,
            dynamic_ncols=True,
        )
        for generation_tokens, prompt_hiddens, generation_hiddens, is_finished in pbar:
            result.prompt_hiddens = prompt_hiddens
            result.generation_tokens_lst.append(generation_tokens)
            result.generation_length_lst.append(len(generation_tokens))
            result.generation_hiddens_lst.append(generation_hiddens)
            result.generation_str_lst.append(self.tokenizer.decode(generation_tokens))

        if prompt_only:
            # Try to maintain compatibility
            result.generation_hiddens_lst = [
                np.zeros((0, *x.shape[1:]), dtype=x.dtype) for x in result.generation_hiddens_lst
            ]

        result.time_cost = toc()
        return result

    @classmethod
    def convert_alpaca_to_prompt(cls, example: AlpacaExample) -> str:
        if example["input"] == "":
            return example["instruction"]
        else:
            return example["instruction"] + "\n" + example["input"]

    def convert_dialog_to_tokens(self, dialog: Dialog) -> list[int]:
        """https://github.com/facebookresearch/llama/blob/7e1b864d574fe6f5ff75fa1d028feb269f7152d2/llama/generation.py#L324-L361"""
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": f'{B_SYS}{dialog[0]["content"]}{E_SYS}{dialog[1]["content"]}',
                },
                *dialog[2:],
            ]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens: list[int] = sum(
            [
                self.tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    bos=True,
                    eos=True,
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += self.tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            bos=True,
            eos=False,
        )
        return dialog_tokens
