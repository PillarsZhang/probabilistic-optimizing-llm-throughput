import atexit
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path
import re
import threading
import traceback
import fire
import lmdb
import random
import numpy as np
import srsly
from loguru import logger
from loguru._logger import Logger as LoguruLogger
import torch
from tqdm import tqdm

from lsdist import __meta__
from lsdist.utils import get_env, get_script_id, get_ver, format_bytes, slices, tic
from lsdist.utils.io import get_lmdb_keys, use_file
from lsdist.utils.log import init_loguru_logger
from lsdist.sample.llama import LlamaSampleResult, LlamaSampleKey
from lsdist.dataset.dist_lmdb import LMDB_MAP_SIZE
from lsdist.dataset.trainable_lmdb import TrainableRecord
from lsdist.utils.numerical import init_random, split_dataset
from lsdist.utils.parallel import MP_CONTEXT, async_executor, watch_parent_kill

DEV = get_env() == "development"
SCRIPT_ID = get_script_id(__file__) + get_ver()
SHUFFLE_SEED = 20231005  # Must same as sample script
SPLIT_SEED = 20231113  # Split is performed on prompt
RECORD_HASH_SALT = b"Yoimiya"
INF_SUPP_ROUND = 1 << 10
INF_DATASET_IDX = 1 << 30
MAX_WORKERS = 4


def main(
    split_props: tuple = (0.8, 0.1, 0.1),
    batch_size: int = 12,
    supp_round_slices: str = "0,1",
    dist_lmdb: str = f"saved/sample/llama_chat/dist_lmdb/supp_round={{supp_round}}",
    num_samples: int = 60,
    prompt_only: bool = True,
    pooling: str = "last",
    trainable_lmdb: str = f"saved/{SCRIPT_ID}/trainable_lmdb/version=5,pooling={{pooling}}",
    dataset_selected_slices: str = ":10000",
    global_log: str = f"saved/{SCRIPT_ID}/global_log/{{timestamp}}_{{log_id}}.log",
):
    init_random()
    init_loguru_logger(use_file(global_log))
    logger.info(f"SCRIPT_ID: {SCRIPT_ID}, locals(): {locals()}")

    #### Parameter

    dataset_selected_slices = slices.parse(dataset_selected_slices)
    dataset_indices = slices.apply(range(INF_DATASET_IDX), dataset_selected_slices)
    if INF_DATASET_IDX in dataset_indices:
        raise ValueError(f"dataset_selected_slices not closed: {dataset_selected_slices}")
    supp_round_slices = slices.parse(supp_round_slices)
    supp_round_tup = slices.apply(range(INF_SUPP_ROUND), supp_round_slices)
    if INF_SUPP_ROUND in supp_round_tup:
        raise ValueError(f"supp_round_slices not closed: {supp_round_slices}")

    trainable_lmdb: Path = use_file(trainable_lmdb.format(pooling=pooling))
    trainable_lmdb_lock = trainable_lmdb.with_name(trainable_lmdb.name + "-lock")
    if trainable_lmdb.exists():
        if DEV:
            logger.warning("Auto rm {trainable_lmdb.as_posix()} (-lock) in development mode!")
            trainable_lmdb.unlink()
            trainable_lmdb_lock.unlink(True)
        else:
            raise Exception(f"Please `rm {trainable_lmdb.as_posix()}*` by yourself first!")
    p = trainable_lmdb.as_posix()
    trainable_lmdb_write_kwargs = dict(path=p, subdir=False, map_size=LMDB_MAP_SIZE)
    logger.debug(f"dist_lmdb_write_kwargs: {trainable_lmdb_write_kwargs}")

    max_num_samples = num_samples * len(supp_round_tup)

    r_env: lmdb.Environment
    w_env: lmdb.Environment

    #### LMDB

    r_envs = ()
    dist_lmdb_read_kwargs_tup = ()
    for supp_round in supp_round_tup:
        p = use_file(dist_lmdb.format(supp_round=supp_round), new=False).as_posix()
        dist_lmdb_read_kwargs = dict(
            path=p,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=LMDB_MAP_SIZE,
        )
        dist_lmdb_read_kwargs_tup += (dist_lmdb_read_kwargs,)
        logger.debug(f"supp_round={supp_round}, dist_lmdb_read_kwargs: {dist_lmdb_read_kwargs}")
        r_env = lmdb.open(**dist_lmdb_read_kwargs)
        atexit.register(r_env.close)
        logger.info(f"Open LMDB {r_env.path()}, stat: {r_env.stat()}")
        r_envs += (r_env,)

    w_env = lmdb.open(**trainable_lmdb_write_kwargs)
    atexit.register(w_env.close)
    logger.info(f"Open LMDB {w_env.path()}, stat: {w_env.stat()}")

    # Check idx key all in LMDB
    for r_env, supp_round in zip(r_envs, supp_round_tup):
        key_bytes_want_set = set(
            LlamaSampleKey(SHUFFLE_SEED, dataset_idx, batch_size, supp_round, num_samples).dumps()
            for dataset_idx in dataset_indices
        )
        logger.info(
            f"supp_round: {supp_round}, key_bytes_want_set: set(length={len(key_bytes_want_set)})"
        )
        with r_env.begin() as txn:
            key_bytes_exist_set = get_lmdb_keys(txn)
        key_bytes_404_set = key_bytes_want_set - key_bytes_exist_set
        logger.info(
            f"{r_env.path()} -> "
            f"key_bytes_exist_set: {len(key_bytes_exist_set)}, "
            f"key_bytes_404_set: {key_bytes_404_set}"
        )
        if key_bytes_404_set:
            raise Exception("There are missing samples!")

    # Check and copy generation_config
    b_lst = []
    for r_env in r_envs:
        with r_env.begin() as txn:
            b_lst.append(txn.get(b"__generation_config__"))
    assert all(b == b_lst[0] for b in b_lst), f"Generation config conflicts: {b_lst}"
    with w_env.begin(write=True) as txn:
        txn.put(b"__generation_config__", b_lst[0])
    logger.debug(f"Copy generation_config to LMDB: {b_lst[0]}")

    #### Split test set

    # Determine the test set here
    split_name_tup = ("train", "valid", "test")
    split_idx_tups = split_dataset(dataset_indices, split_props, SPLIT_SEED)
    split_idx_dic = dict(zip(split_name_tup, split_idx_tups))
    split_idx_reversed_dic = {vv: k for k, v in split_idx_dic.items() for vv in v}
    d = {k: f"{v.__class__.__name__}(length={len(v)})" for k, v in split_idx_dic.items()}
    logger.info(f"split_idx_dic: {d}")

    # Put split_idx to LMDB
    with w_env.begin(write=True) as txn:
        txn.put(b"__split_idx__", srsly.json_dumps(split_idx_dic).encode())
    logger.debug("Put split_idx_dic to LMDB")

    # Put max_num_samples to LMDB
    with w_env.begin(write=True) as txn:
        txn.put(b"__max_num_samples__", srsly.json_dumps(max_num_samples).encode())
    logger.debug(f"Put max_num_samples={max_num_samples} to LMDB")

    #### Convert

    idx_to_h_dic: dict[int, set] = dict()
    idx_to_prompt_h_dic: dict[int, set] = dict()

    executor = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=MP_CONTEXT,
        initializer=_pool_initializer_lmdb,
        initargs=(dist_lmdb_read_kwargs_tup, trainable_lmdb_write_kwargs, logger),
    )

    submit_pbar = tqdm(dataset_indices, desc="Converting -> submit", dynamic_ncols=True)
    work_pbar = tqdm(dataset_indices, desc="Converting -> work", dynamic_ncols=True)

    def on_success(result, func, *args, **kwargs):
        if result:
            work_pbar.update()

    num_executor_error = 0

    def on_error(error, func, *args, **kwargs):
        nonlocal num_executor_error
        num_executor_error += 1
        traceback.print_exception(error)
        logger.error(
            f"Executor error happen ({num_executor_error}) - "
            f"error: {error}, func: {func}, args: {args}, kwargs: {kwargs}"
        )

    convert = async_executor(
        executor, on_success=on_success, on_error=on_error, callback_input=True
    )(_pool_function_convert)
    convert(None, None, None, ping=True)

    future_lst = []
    for dataset_idx in submit_pbar:
        key_bytes_tup = tuple(
            LlamaSampleKey(SHUFFLE_SEED, dataset_idx, batch_size, supp_round, num_samples).dumps()
            for supp_round in supp_round_tup
        )
        future_lst.append(convert(dataset_idx, key_bytes_tup, prompt_only, pooling))
    concurrent.futures.wait(future_lst)
    work_pbar.close()

    if num_executor_error > 0:
        logger.error(f"Found {num_executor_error} executor error happen!")
        raise Exception("executor error happen")
    else:
        logger.success("No executor error happen!")

    collect_pbar = tqdm(dataset_indices, desc="Converting -> collect", dynamic_ncols=True)
    for dataset_idx, future in zip(collect_pbar, future_lst):
        idx_to_h_dic[dataset_idx], idx_to_prompt_h_dic[dataset_idx] = future.result()
    executor.shutdown()

    split_h_dic = {k: set.union(*(idx_to_h_dic[vv] for vv in v)) for k, v in split_idx_dic.items()}
    len_dic = {k: len(v) for k, v in split_h_dic.items()}
    logger.info(f"split_h_dic: {len_dic}")

    # Put split_h to LMDB
    with w_env.begin(write=True) as txn:
        txn.put(b"__split_h__", srsly.pickle_dumps(split_h_dic))
    logger.debug("Put split_h_dic to LMDB")

    # Put idx_to_h, idx_to_prompt_h to LMDB
    with w_env.begin(write=True) as txn:
        txn.put(b"__idx_to_h__", srsly.pickle_dumps(idx_to_h_dic))
        txn.put(b"__idx_to_prompt_h__", srsly.pickle_dumps(idx_to_prompt_h_dic))
    logger.debug("Put idx_to_h_dic, idx_to_prompt_h_dic to LMDB")

    logger.success(f"[{SCRIPT_ID}] done!")


_pool_data = threading.local()


def _pool_initializer_lmdb(
    lmdb_read_kwargs_tup: tuple[dict], lmdb_write_kwargs: dict, logger: LoguruLogger
):
    watch_parent_kill()
    atexit.register(logger.complete)

    def format_env(env):
        return (
            f"pid: {os.getpid()}, tid: {threading.get_ident()}, "
            f"env: {id(env)}, stat: {env.stat()}"
        )

    def clean_env(env):
        logger.info(f"Close LMDB {env.path()}, {format_env(env)}")
        env.close()

    r_envs = ()
    for lmdb_read_kwargs in lmdb_read_kwargs_tup:
        r_env = lmdb.open(**lmdb_read_kwargs)
        atexit.register(clean_env, r_env)
        logger.info(f"Open LMDB {r_env.path()}, {format_env(r_env)}")
        r_envs += (r_env,)

    w_env = lmdb.open(**lmdb_write_kwargs)
    logger.info(f"Open LMDB {w_env.path()}, {format_env(w_env)}")
    atexit.register(clean_env, w_env)

    _pool_data.r_envs = r_envs
    _pool_data.w_env = w_env
    _pool_data.logger = logger


def _pool_function_convert(
    dataset_idx: int,
    key_bytes_tup: tuple[bytes],
    prompt_only: bool,
    pooling: str = "last",
    ping: bool = False,
):
    r_envs: lmdb.Environment = _pool_data.r_envs
    w_env: lmdb.Environment = _pool_data.w_env
    logger: LoguruLogger = _pool_data.logger

    if ping:
        with w_env.begin(write=True) as txn:
            txn.put(b"__meta__", srsly.json_dumps(__meta__).encode())
        logger.success(f"LMDB works fine!")
        return None

    len_lst = []
    time_lst = []
    result_lst = []
    h_set = set()
    prompt_h_set = set()
    toc = tic()
    for r_env, key_bytes in zip(r_envs, key_bytes_tup):
        with r_env.begin() as txn:
            result_bytes = txn.get(key_bytes)
        result = LlamaSampleResult.loads(result_bytes)
        result_lst.append(result)
    time_lst.append(toc())

    if prompt_only:
        record_lst = prompt_result(result_lst, dataset_idx, pooling)
    else:
        # Scan first
        record_lst = scan_result(result_lst, dataset_idx, pooling)
        len_lst.append(len(record_lst))
        # Then filter by policy
        record_lst = filter_policy(record_lst, dataset_idx)
    len_lst.append(len(record_lst))
    time_lst.append(toc())

    # Save to LMDB
    r_bytes_sum_length = 0
    with w_env.begin(write=True) as txn:
        for r in record_lst:
            r_bytes = r.dumps()
            r_bytes_sum_length += len(r_bytes)
            h = hashlib.sha256(r_bytes + RECORD_HASH_SALT)
            h_bytes = h.digest()
            txn.put(h_bytes, r_bytes)
            h_set.add(h_bytes)
            if "prompt" in r.tags:
                prompt_h_set.add(h_bytes)

    time_lst.append(toc())
    time_lst = [f"{x:.3f}" for x in time_lst]

    logger.debug(
        f"[{dataset_idx}] "
        f"size: {format_bytes(r_bytes_sum_length)}, "
        f"time_lst: ({', '.join(time_lst)}) secs, "
        f"accumulate: {len(record_lst)}, "
        f"filter: {len_lst}"
    )
    return h_set, prompt_h_set


def apply_pooling(hiddens: np.ndarray, pooling: str):
    if pooling == "last":
        hidden = torch.tensor(hiddens[-1]).bfloat16()
        assert hidden.shape[0] == hiddens.shape[-1]
    elif pooling == "mean":
        hidden = torch.tensor(hiddens.mean(0)).bfloat16()
        assert hidden.shape[0] == hiddens.shape[-1]
    elif match := re.match(r"^last(\d+)$", pooling):
        n = int(match.group(1))
        if hiddens.shape[0] < n:
            padded_hiddens = np.zeros((n, hiddens.shape[-1]))
            padded_hiddens[-hiddens.shape[0] :] = hiddens
            hiddens = padded_hiddens
        hidden = torch.tensor(hiddens[-n:].reshape(-1)).bfloat16()
        assert hidden.shape[0] == hiddens.shape[-1] * n
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    return hidden


def prompt_result(result_lst: list[LlamaSampleResult], dataset_idx: int, pooling: str = "last"):
    result = LlamaSampleResult.sum(result_lst)
    hiddens = result.prompt_hiddens[-1]
    hidden = apply_pooling(hiddens, pooling)

    r = TrainableRecord(
        tokens=result.prompt_tokens,
        hidden=hidden,
        prev_length=result.prompt_length,
        post_lengths=result.generation_length_lst,
        tags=("prompt",),
        dataset_idx=dataset_idx,
    )
    return [r]


def scan_result(result_lst: list[LlamaSampleResult], dataset_idx: int, pooling: str = None):
    result = LlamaSampleResult.sum(result_lst)
    record_dic = dict()
    generation_zip = zip(result.generation_tokens_lst, result.generation_hiddens_lst)
    for generation_tokens, generation_hiddens in generation_zip:
        for trunc_idx in range(len(generation_tokens)):
            tokens = tuple(result.prompt_tokens + generation_tokens[:trunc_idx])
            if tokens in record_dic:
                r = record_dic[tokens]
                r.post_lengths.append(len(generation_tokens) - trunc_idx)
            else:
                hiddens = np.concatenate(
                    (result.prompt_hiddens[-1, :], generation_hiddens[-1, :trunc_idx]), axis=0
                )
                hidden = apply_pooling(hiddens, pooling)
                r = TrainableRecord(
                    tokens=tokens,
                    hidden=hidden,
                    prev_length=len(tokens),
                    post_lengths=[len(generation_tokens) - trunc_idx],
                    tags=("prompt",) if trunc_idx == 0 else (),
                    dataset_idx=dataset_idx,
                )
                record_dic[tokens] = r
    return list(record_dic.values())


def filter_policy(record_lst: list[TrainableRecord], dataset_idx: int):
    # 1. Select 256 with the highest number of lengths
    # 2. Randomly select 256 from the remaining
    pol_1, pol_2 = 256, 256
    record_lst_sorted = sorted(
        record_lst, key=lambda r: (len(r.post_lengths), r.tokens), reverse=True
    )
    lst_1, lst_2 = record_lst_sorted[:pol_1], record_lst_sorted[pol_1:]
    lst_2 = random.Random(dataset_idx).sample(lst_2, min(pol_2, len(lst_2)))
    return lst_1 + lst_2


if __name__ == "__main__":
    fire.Fire(main)
