import atexit
import os
import threading
import fire
import numpy as np
import srsly
import lmdb
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from loguru import logger
from loguru._logger import Logger as LoguruLogger

from lsdist import __meta__
from lsdist.utils import get_script_id, get_ver, format_bytes, tic
from lsdist.utils import slices
from lsdist.llama import Llama
from lsdist.utils.distributed import on_rank
from lsdist.utils.io import get_lmdb_keys, use_file
from lsdist.utils.data import use_hf_dataset
from lsdist.utils.log import init_loguru_logger
from lsdist.utils.numerical import init_random
from lsdist.sample.llama import (
    LlamaDistSampler,
    LlamaGenerationConfig,
    LlamaSampleKey,
    LlamaSampleResult,
)
from lsdist.utils.parallel import MP_CONTEXT, async_executor
from lsdist.dataset.dist_lmdb import LMDB_MAP_SIZE

SCRIPT_ID = get_script_id(__file__) + get_ver()
SHUFFLE_SEED = 20231005
MAX_WORKERS = 8


def main(
    llama_ckpt_dir: str = "../llama/llama-2-7b-chat",
    llama_tokenizer_model: str = "../llama/tokenizer.model",
    generation_config_yml: str = "scripts/sample/generation_config.yml",
    enable_bfloat16: bool = True,
    hidden_saved_slices: str = "-1",
    async_save_method: str = "process",
    skip_exist: bool = True,
    batch_size: int = 12,
    supp_round: int = 0,
    num_samples: int = 60,
    prompt_only: bool = True,
    dataset_selected_slices: str = ":",
    dataset_name: str = "tatsu-lab/alpaca",
    dataset_offline: bool = True,
    huggingface_offline_cache_dir: str = "~/.cache/huggingface_offline_cache",
    dist_lmdb: str = f"saved/{SCRIPT_ID}/dist_lmdb/supp_round={{supp_round}}",
    global_log: str = f"saved/{SCRIPT_ID}/global_log/{{timestamp}}_{{log_id}}.log",
):
    init_random()
    init_loguru_logger(use_file(global_log))
    logger.info(f"SCRIPT_ID: {SCRIPT_ID}, locals(): {locals()}")

    #### Parameter

    huggingface_offline_cache_dir = Path(huggingface_offline_cache_dir)
    hidden_saved_slices = slices.parse(hidden_saved_slices)
    dataset_selected_slices = slices.parse(dataset_selected_slices)
    assert (async_save_method in {"thread", "process"}) or not async_save_method

    dist_lmdb: Path = use_file(dist_lmdb.format(supp_round=supp_round))
    p = dist_lmdb.as_posix()
    dist_lmdb_read_kwargs = dict(path=p, subdir=False, readonly=True, map_size=LMDB_MAP_SIZE)
    dist_lmdb_write_kwargs = dict(path=p, subdir=False, map_size=LMDB_MAP_SIZE)
    logger.debug(f"dist_lmdb_read_kwargs: {dist_lmdb_read_kwargs}")
    logger.debug(f"dist_lmdb_write_kwargs: {dist_lmdb_write_kwargs}")

    env: lmdb.Environment

    #### Dataset

    if dataset_name == "tatsu-lab/alpaca":
        dataset = use_hf_dataset(
            dataset_name, dataset_offline, huggingface_offline_cache_dir / "dataset"
        )
        dataset = dataset["train"].shuffle(SHUFFLE_SEED)
    else:
        raise f"Unknown dataset: {dataset_name}"

    dataset_indices = slices.apply(range(len(dataset)), dataset_selected_slices)
    logger.info(
        f"Filter dataset_selected_slices {slices.format(dataset_selected_slices)} -> {len(dataset_indices)} examples"
    )

    if skip_exist and dist_lmdb.exists():
        logger.info(f"Filter existing keys because skip_exist={skip_exist}")
        with lmdb.open(**dist_lmdb_read_kwargs) as env:
            with env.begin() as txn:
                key_bytes_exist_set = get_lmdb_keys(txn)
        logger.debug(f"Found {len(key_bytes_exist_set)} keys in lmdb")

        dataset_indices_want = []
        for dataset_idx in dataset_indices:
            key_bytes_want = LlamaSampleKey(
                SHUFFLE_SEED, dataset_idx, batch_size, supp_round, num_samples
            ).dumps()
            if key_bytes_want not in key_bytes_exist_set:
                dataset_indices_want.append(dataset_idx)
        dataset_indices = tuple(dataset_indices_want)
        logger.info(f"Filter skip_exist -> {len(dataset_indices)} examples")

    dataset = dataset.select(dataset_indices)
    assert len(dataset) == len(dataset_indices)

    logger.info(repr(repr(dataset)))
    logger.success(f"Init {dataset_name} dataset done!")

    #### Model

    generation_config_dic: LlamaGenerationConfig = srsly.read_yaml(
        use_file(generation_config_yml, new=False)
    )
    logger.info(f"generation_config_dic: {generation_config_dic}")

    # Check config in generation_config LMDB
    b_want = srsly.json_dumps(generation_config_dic).encode()
    logger.debug(f"b_want: {b_want}")
    with lmdb.open(**dist_lmdb_write_kwargs) as env:
        with env.begin() as txn:
            b_exist = txn.get(b"__generation_config__")
        logger.debug(f"b_exist: {b_exist}")
        if b_exist is None:
            with env.begin(write=True) as txn:
                txn.put(b"__generation_config__", b_want)
            b_exist = b_want
            logger.debug("Put new config to LMDB")
    if b_exist == b_want:
        logger.success("Check generation_config in LMDB: pass!")
    else:
        logger.error("Check generation_config in LMDB: conflict!")
        raise Exception(f"generation_config conflict!")

    generator = Llama.build(
        ckpt_dir=llama_ckpt_dir,
        tokenizer_path=llama_tokenizer_model,
        max_seq_len=generation_config_dic["max_seq_len"],
        max_batch_size=batch_size,
        enable_bfloat16=enable_bfloat16,
    )
    hidden_indices = slices.apply(range(generator.model.params.n_layers), hidden_saved_slices)
    logger.info(f"Save these hiddens: {hidden_indices}")

    #### Saver

    if on_rank():
        if async_save_method:
            if async_save_method == "thread":
                executor = ThreadPoolExecutor(
                    max_workers=MAX_WORKERS,
                    thread_name_prefix="LsdistSaver-ThreadPoolExecutor",
                    initializer=_pool_initializer_lmdb,
                    initargs=(dist_lmdb_write_kwargs, logger),
                )
            else:
                executor = ProcessPoolExecutor(
                    max_workers=MAX_WORKERS,
                    mp_context=MP_CONTEXT,
                    initializer=_pool_initializer_lmdb,
                    initargs=(dist_lmdb_write_kwargs, logger),
                )
            saver = async_executor(executor)(_pool_function_saver)
        else:
            _pool_initializer_lmdb(dist_lmdb_write_kwargs)
            saver = _pool_function_saver

        # Ping saver
        saver(None, None, ping=True)

    #### Sampler

    sampler = LlamaDistSampler(generator)

    for idx, example in enumerate(dataset):
        example_desc = f"[{idx}/{len(dataset)}, {dataset_indices[idx]}/{slices.format(dataset_selected_slices)}]"
        prompt = sampler.convert_alpaca_to_prompt(example)
        logger.info(f"Sample example {example_desc} start, prompt: {repr(prompt)}")
        key = LlamaSampleKey(
            SHUFFLE_SEED, dataset_indices[idx], batch_size, supp_round, num_samples
        )
        result = sampler.sample_from_prompt(
            prompt,
            temperature=generation_config_dic["temperature"],
            top_p=generation_config_dic["top_p"],
            max_gen_len=generation_config_dic["max_gen_len"],
            supp_round=supp_round,
            num_samples=num_samples,
            hidden_indices=hidden_indices,
            prompt_only=prompt_only,
        )
        if on_rank():
            logger.info(
                f"Sample example {example_desc} done, "
                f"generation_length: (mean) {np.mean(result.generation_length_lst):.3f}, "
                f"generate_time_cost: {result.time_cost:.3f} secs -> {result.time_cost / num_samples:.3f} secs/sample"
            )
            saver(key, result)

    if on_rank():
        if async_save_method:
            executor.shutdown()

    logger.success(f"[{SCRIPT_ID}] done!")


_pool_data = threading.local()


def _pool_initializer_lmdb(lmdb_write_kwargs: dict, logger: LoguruLogger):
    env: lmdb.Environment = lmdb.open(**lmdb_write_kwargs)

    def format_env(env):
        return (
            f"pid: {os.getpid()}, tid: {threading.get_ident()}, "
            f"env: {id(env)}, stat: {env.stat()}"
        )

    logger.info(f"Open LMDB {env.path()}, {format_env(env)}")

    def cleaner():
        logger.info(f"Close LMDB {env.path()}, {format_env(env)}")
        logger.complete()
        env.close()

    atexit.register(cleaner)
    _pool_data.env = env
    _pool_data.logger = logger


def _pool_function_saver(key: LlamaSampleKey, result: LlamaSampleResult, ping: bool = False):
    env: lmdb.Environment = _pool_data.env
    logger: LoguruLogger = _pool_data.logger

    if ping:
        with env.begin(write=True) as txn:
            txn.put(b"__meta__", srsly.json_dumps(__meta__).encode())
        logger.success(f"LMDB works fine!")
        return

    toc = tic()
    logger.debug(f"Save result {repr(key)} start")

    key_bytes = key.dumps()
    result_bytes = result.dumps()
    with env.begin(write=True) as txn:
        txn.put(key_bytes, result_bytes)

    time_cost = toc()
    logger.debug(
        f"Save result {repr(key)} done, "
        f"key_bytes: {repr(key_bytes)}, "
        f"result_bytes: bytes(size={format_bytes(len(result_bytes))}), "
        f"dump_time_cost: {time_cost:.3f} secs -> {format_bytes(len(result_bytes) / time_cost)}/sec"
    )


if __name__ == "__main__":
    fire.Fire(main)
