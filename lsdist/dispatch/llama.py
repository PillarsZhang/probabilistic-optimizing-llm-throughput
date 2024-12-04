import math
from loguru import logger
import srsly
from typing import TypedDict
from more_itertools import chunked
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from tqdm import tqdm

from ..llama import Llama, Dialog
from ..llama.generation import B_INST, E_INST, B_SYS, E_SYS

from ..utils.numerical import sorted_with_indices
from ..utils.data import AlpacaExample, Dialog
from ..utils import tic


class LlamaGenerationConfig(TypedDict):
    temperature: float
    top_p: float
    max_seq_len: int
    max_gen_len: int | None


class LlamaDistDispatcher:
    def __init__(self, generator: Llama, dist_model: nn.Module, dev: bool = False) -> None:
        self.generator = generator
        self.model = generator.model
        self.tokenizer = generator.tokenizer

        self.dist_model = dist_model
        self.dev = dev

    def sample_from_chunk(
        self,
        example_chunk: list[str],
        dist_batch_size: int,
        gen_batch_size: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: int = None,
        hidden_indices: tuple = (),
        batched_rule: tuple = (),
        known_lengths: list[int] = None,
        known_lengths_tc: float = None,
        model_scalar: bool = False,
    ):
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        time_history = dict()
        other_info = dict()
        toc = tic()

        prompt_chunk = self.make_prompt_chunk(example_chunk)
        time_history["make_prompt_chunk"] = toc()

        prompt_length_chunk = np.array(list(map(len, prompt_chunk)), dtype=int)
        fcr = "fcr" in batched_rule

        if not batched_rule or batched_rule[0] == "vanilla":
            ## No boost
            new_indexes_batch: list[list[int]] = list(
                chunked(range(len(prompt_chunk)), gen_batch_size)
            )
            max_length_batch = None
            predicted_lengths = None
        else:
            if known_lengths is None:
                ## Predict distribution
                # sorted_indices, sorted_chunk = sorted_with_indices(prompt_chunk, key=len)
                # dist_chunk = []
                # self.dist_model.eval()
                # num_batchs = math.ceil(len(sorted_chunk) / dist_batch_size)

                # for prompt_batch in tqdm(
                #     chunked(sorted_chunk, dist_batch_size),
                #     total=num_batchs,
                #     desc="Predict distribution",
                # ):
                #     hidden = self.generator.embed(prompt_batch, hidden_indices)
                #     with torch.no_grad():
                #         dist_batch: Tensor = self.dist_model(hidden)
                #     dist_chunk.append(dist_batch)
                # dist_chunk: NDArray = torch.cat(dist_chunk, dim=0).softmax(dim=1).cpu().numpy()
                # dist_chunk[sorted_indices] = dist_chunk

                ## Predict distribution
                dist_chunk = []
                self.dist_model.eval()
                num_batches = math.ceil(len(prompt_chunk) / dist_batch_size)

                for prompt_batch in tqdm(
                    chunked(prompt_chunk, dist_batch_size),
                    total=num_batches,
                    desc="Predict distribution",
                ):
                    hidden = self.generator.embed(prompt_batch, hidden_indices)
                    with torch.no_grad():
                        dist_batch: Tensor = self.dist_model(hidden)
                    dist_chunk.append(dist_batch)
                if model_scalar:
                    dist_chunk = torch.cat(dist_chunk, dim=0).cpu().numpy()
                else:
                    dist_chunk = torch.cat(dist_chunk, dim=0).softmax(dim=1).cpu().numpy()

                ## Batched again
                new_indexes_batch = self.batched_again(
                    dist_chunk,
                    prompt_length_chunk,
                    gen_batch_size,
                    batched_rule=batched_rule,
                    model_scalar=model_scalar,
                )

                if batched_rule[0] == "search":
                    best_th = new_indexes_batch[0][2]
                    best_l = new_indexes_batch[0][3]
                    predicted_lengths = (np.cumsum(dist_chunk, axis=1) >= best_th).argmax(axis=1)
                    logger.debug(f"{best_th=}, {best_l=}")
                else:
                    if model_scalar:
                        predicted_lengths = dist_chunk.flatten() + prompt_length_chunk
                    else:
                        predicted_lengths = (
                            np.cumsum(dist_chunk, axis=1) >= batched_rule[1]
                        ).argmax(axis=1)

            else:
                ## Use known lengths
                ## Batched again

                new_indexes_batch = self.batched_again(
                    None,
                    prompt_length_chunk,
                    gen_batch_size,
                    batched_rule=batched_rule,
                    known_lengths=known_lengths,
                )
                predicted_lengths = np.array(known_lengths, dtype=int) + prompt_length_chunk
            other_info["predicted_lengths"] = predicted_lengths
            logger.debug(f"{prompt_length_chunk=}, {predicted_lengths=}")

            time_history["batched_again"] = toc()

            raw_new_indexes_batch = new_indexes_batch
            other_info["raw_new_indexes_batch"] = raw_new_indexes_batch
            if type(new_indexes_batch[0][0]) in (tuple, list):
                max_length_batch: list[int] = [x[1] for x in new_indexes_batch]
                new_indexes_batch: list[list[int]] = [x[0] for x in new_indexes_batch]
            else:
                max_length_batch = None

            other_info["new_indexes_batch"] = new_indexes_batch
            other_info["max_length_batch"] = max_length_batch

            if fcr and max_length_batch is None:
                raise ValueError(f"max_length_batch is None, {fcr=}")

            if known_lengths_tc is not None:
                time_history["batched_again"] += known_lengths_tc

        # Assert sorted chain new_indexes_batch same as range
        assert sorted(sum(new_indexes_batch, [])) == list(range(len(prompt_chunk)))

        # Generate
        generation_tokens = np.empty(len(prompt_chunk), dtype=object)
        generation_strings = np.empty(len(prompt_chunk), dtype=object)
        is_finished_lst = np.empty(len(prompt_chunk), dtype=object)

        generation_num_tokens = 0
        generation_num_lengths = 0
        for i in tqdm(
            range(len(new_indexes_batch)), desc=f"Generate {batched_rule=}, {temperature=}"
        ):
            new_indexes = new_indexes_batch[i]
            prompt_batch = prompt_chunk[new_indexes]
            max_length_batch_i = (
                min(int(np.ceil(max_length_batch[i])), self.model.params.max_seq_len)
                if max_length_batch is not None
                else None
            )

            logger.debug(f"{new_indexes=}, {len(prompt_batch)=}, {max_length_batch_i=}")
            prompt_lengths = list(map(len, prompt_batch))
            logger.debug(f"{prompt_lengths=}")
            if predicted_lengths is not None:
                logger.debug(f"{predicted_lengths[new_indexes]=}")
            if fcr:
                self.model.reset_cache(
                    len(prompt_batch),
                    min(max_length_batch_i + 8, self.model.params.max_seq_len),
                )
            _generation_tokens, *_, _is_finished_lst = self.generator.generate(
                prompt_tokens=prompt_batch,
                # max_gen_len=max_gen_len if not fcr else max_length_batch_i,
                max_gen_len=max_gen_len,
                max_seq_len=None if not fcr else max_length_batch_i,
                temperature=temperature,
                top_p=top_p,
            )
            generation_lengths = list(map(len, _generation_tokens))
            generation_num_tokens += sum(generation_lengths)
            generation_num_lengths += max(generation_lengths) * len(generation_lengths)
            # logger.debug(
            #     f"{generation_lengths=}, {max(generation_lengths)=}, {np.mean(generation_lengths)=}, {len(generation_lengths)=}"
            # )
            sentence_lengths = np.array(generation_lengths) + np.array(list(map(len, prompt_batch)))
            sentence_lengths = sentence_lengths.tolist()
            logger.debug(
                f"{sentence_lengths=}, {max(sentence_lengths)=}, {np.mean(sentence_lengths)=}, {len(sentence_lengths)=}"
            )
            generation_tokens[new_indexes] = _generation_tokens
            generation_strings[new_indexes] = self.tokenizer.decode(_generation_tokens)
            is_finished_lst[new_indexes] = _is_finished_lst

        time_history["generate"] = toc()

        # Re-generate

        not_finished_indexes = np.where(np.logical_not(is_finished_lst))[0]

        other_info["generation_tokens_before_fcr"] = np.copy(generation_tokens)
        other_info["generation_strings_before_fcr"] = np.copy(generation_strings)
        other_info["not_finished_indexes"] = np.copy(not_finished_indexes)

        if len(not_finished_indexes) > 0:
            logger.warning(
                f"{len(not_finished_indexes)} examples are not finished. {not_finished_indexes.tolist()=}"
            )
            not_finished_indexes = [
                i
                for i in not_finished_indexes
                if len(prompt_chunk[i]) + len(generation_tokens[i])
                < self.model.params.max_seq_len - 8
            ]
            logger.info(f"{len(not_finished_indexes)} examples are not overflow after filter.")
            if fcr and len(not_finished_indexes) > 0:
                self.model.reset_cache(
                    self.model.params.max_batch_size, self.model.params.max_seq_len
                )
                logger.info(f"Re-generate {batched_rule=}")
                for new_indexes in tqdm(
                    list(chunked(not_finished_indexes, gen_batch_size)),
                    desc=f"Re-generate, {batched_rule=}, {temperature=}",
                ):
                    prompt_batch = [
                        x + y
                        for x, y in zip(prompt_chunk[new_indexes], generation_tokens[new_indexes])
                    ]
                    _generation_tokens, *_, _is_finished_lst = self.generator.generate(
                        prompt_tokens=prompt_batch,
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    generation_lengths = list(map(len, _generation_tokens))
                    generation_num_tokens += sum(generation_lengths)
                    generation_num_lengths += max(generation_lengths) * len(generation_lengths)
                    logger.debug(
                        f"FCR | {generation_lengths=}, {max(generation_lengths)=}, {np.mean(generation_lengths)=}, {len(generation_lengths)=}"
                    )
                    generation_tokens[new_indexes] = [
                        x + y for x, y in zip(generation_tokens[new_indexes], _generation_tokens)
                    ]
                    generation_strings[new_indexes] = self.tokenizer.decode(
                        list(generation_tokens[new_indexes])
                    )
                    is_finished_lst[new_indexes] = _is_finished_lst

                not_finished_indexes = np.where(np.logical_not(is_finished_lst))[0]
                if len(not_finished_indexes) > 0:
                    logger.warning(
                        f"FCR | {len(not_finished_indexes)} examples are not finished. {not_finished_indexes=}"
                    )

        time_history["re-generate"] = toc()

        return (
            generation_tokens,
            generation_strings,
            time_history,
            generation_num_tokens,
            generation_num_lengths,
            other_info,
        )

    def batched_again(
        self,
        pdf: NDArray,
        d: NDArray,
        batch_size: int,
        batched_rule: tuple,
        known_lengths: list[int] = None,
        model_scalar: bool = False,
    ):
        method, *args = batched_rule
        if method == "cdf":
            (threshold,) = args
            cdf = np.cumsum(pdf, axis=1)
            x: NDArray = (cdf >= threshold).argmax(axis=1)
            sorted_indices = x.argsort()
            chunks = list(chunked(sorted_indices, batch_size))
            return [(chunk, x[chunk].max()) for chunk in chunks]
        elif method == "ward":
            perf_json: str = f"saved/dispatch/eval_performance/llama-2-7b-chat_perf.json"
            with open(perf_json, "r") as f:
                perf_lst = srsly.json_loads(f.read())

            from .ward_test import ward, get_time_cost_map

            time_cost_map = get_time_cost_map(perf_lst, batch_size, 1024)
            return list(ward(time_cost_map, pdf, d))
        elif method == "exchange":
            (threshold,) = args
            perf_json: str = f"saved/dispatch/eval_performance/llama-2-7b-chat_perf.json"
            with open(perf_json, "r") as f:
                perf_lst = srsly.json_loads(f.read())

            from .exchange_test import exchange, get_time_cost_map

            time_cost_map = get_time_cost_map(perf_lst, batch_size, 1023)
            return list(exchange(time_cost_map, pdf, d, batch_size, th=threshold))
        elif method == "dump":
            (fn,) = args
            with open(fn, "wb") as f:
                f.write(srsly.pickle_dumps((pdf, d)))
            exit()
        elif method == "ssch":
            from ..ssch.llama2_utils import schedule as ssch_schedule

            threshold, cell, *_ = args
            vbs = "vbs" in args

            if known_lengths is not None:
                lengths = known_lengths
                lengths = np.array(lengths, dtype=int) + d
            elif model_scalar:
                lengths = pdf.flatten()
                lengths = np.where(lengths < d + 1, d + 1, lengths)
            else:
                cdf = np.cumsum(pdf, axis=1)
                lengths = (cdf >= threshold).argmax(axis=1)

                # Assert length large than prompt
                lengths = np.where(lengths < d + 1, d + 1, lengths)

                # lengths = lengths - d
                # lengths[lengths < 1] = 1
            batches = ssch_schedule(lengths=lengths, mini_batch_size=batch_size, vbs=vbs, cell=cell)
            return batches
        elif method == "search":
            cell, *_ = args
            vbs = "vbs" in args
            fcr = "fcr" in args
            batches = search_ssch_schedule(
                pdfs=pdf, mini_batch_size=batch_size, vbs=vbs, cell=cell, fcr=fcr, d=d
            )
            return batches
        else:
            raise ValueError(f"Unknow {batched_rule=}")

    def make_prompt_chunk(self, example_chunk: list[str]):
        prompt_chunk = []
        for example in example_chunk:
            dialog: Dialog = [
                {
                    "role": "user",
                    "content": self.convert_alpaca_to_prompt(example),
                }
            ]
            dialog_tokens: list[int] = self.convert_dialog_to_tokens(dialog)
            prompt_chunk.append(dialog_tokens)
        prompt_chunk: NDArray = np.array(prompt_chunk, dtype=object)
        return prompt_chunk

    def convert_alpaca_to_prompt(self, example: AlpacaExample) -> str:
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

    def dist_from_chunk(
        self, example_chunk: list[str], dist_batch_size: int, hidden_indices: tuple = ()
    ):
        prompt_chunk = self.make_prompt_chunk(example_chunk)
        prompt_length_chunk = np.array(list(map(len, prompt_chunk)), dtype=int)

        ## Predict distribution
        dist_chunk = []
        self.dist_model.eval()
        num_batches = math.ceil(len(prompt_chunk) / dist_batch_size)

        for prompt_batch in tqdm(
            chunked(prompt_chunk, dist_batch_size),
            total=num_batches,
            desc="Predict distribution",
        ):
            hidden = self.generator.embed(prompt_batch, hidden_indices)
            with torch.no_grad():
                dist_batch: Tensor = self.dist_model(hidden)
            dist_chunk.append(dist_batch)
        dist_chunk = torch.cat(dist_chunk, dim=0).softmax(dim=1).cpu().numpy()
        return dist_chunk, prompt_length_chunk


def search_ssch_schedule(pdfs, mini_batch_size, vbs, cell, fcr, d):
    from ..ssch.llama2_utils import schedule as ssch_schedule
    from .strategy_sim import sim_fcr, sim_vbs

    threshold_lst = np.linspace(0.10, 0.99, 90)
    cdfs = np.cumsum(pdfs, axis=1)
    max_length = pdfs.shape[1] - 1

    batches_lst = []
    round_length_lst = []
    for threshold in threshold_lst:
        lengths = (cdfs >= threshold).argmax(axis=1)
        lengths = np.where(lengths < d + 1, d + 1, lengths)
        batches = ssch_schedule(
            lengths=lengths, mini_batch_size=mini_batch_size, vbs=vbs, cell=cell
        )
        batches_lst.append(batches)
        _batches = [x[0] for x in batches]
        _stops = [min(x[1], max_length - 1) for x in batches]
        if fcr:
            chunk_pdf, chunk_mean, fcr_num_batch, fcr_mean = sim_fcr(
                pdfs, _batches, _stops, fcr_batch_size=mini_batch_size
            )
            round_length = chunk_mean + fcr_mean
        else:
            chunk_pdf, chunk_mean = sim_vbs(pdfs, _batches)
            round_length = chunk_mean
        round_length_lst.append(round_length)
    best_idx = np.argmin(round_length_lst)
    best_batches = batches_lst[best_idx]
    best_batches = [
        (x[0], x[1], float(threshold_lst[best_idx]), float(round_length_lst[best_idx]))
        for x in best_batches
    ]
    return best_batches
