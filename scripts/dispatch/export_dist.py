import math
from pathlib import Path
import fire
import srsly
from loguru import logger
from more_itertools import chunked

import torch

from lsdist.llama import Llama
from lsdist.dispatch.llama import LlamaDistDispatcher
import lsdist.model.backbone as backbone_module
from lsdist.model.net import DiscreteDistributionNet
from lsdist.sample.llama import LlamaGenerationConfig
from lsdist.utils import get_env, get_script_id, get_ver, slices, tic
from lsdist.utils.data import use_hf_dataset
from lsdist.utils.io import use_dir, use_file
from lsdist.utils.log import init_loguru_logger
from lsdist.utils.numerical import init_random

DEV = get_env() == "development"
DEV_SUFFIX = "__DEV" if DEV else ""
SCRIPT_ID = get_script_id(__file__) + get_ver()
TRAIN_SCRIPT_ID = "train/pt_train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHUFFLE_SEED = 20231005


def main(
    chunk_size: int = 512,
    dist_batch_size: int = 32,
    llama_ckpt_dir: str = "../llama/llama-2-7b-chat",
    llama_tokenizer_model: str = "../llama/tokenizer.model",
    generation_config_yml: str = "scripts/sample/generation_config.yml",
    enable_bfloat16: bool = True,
    dataset_selected_slices: str = "10000:10064",
    dataset_name: str = "tatsu-lab/alpaca",
    dataset_offline: bool = True,
    huggingface_offline_cache_dir: str = "~/.cache/huggingface_offline_cache",
    pt_ckpt: str = f"saved/{TRAIN_SCRIPT_ID}/pt_ckpt",
    tag: str = "20230115C-data-noise-1.0>epoch=204,train_loss=0.239903,valid_loss=0.319597",
    result_pkl: str = f"saved/{SCRIPT_ID}/{{dataset_selected_slices_text}}.pkl",
    global_log: str = f"saved/{SCRIPT_ID}/global_log/{{timestamp}}_{{log_id}}.log",
    new_temperature: float = None,
):
    init_random()
    init_loguru_logger(use_file(global_log))
    logger.info(f"SCRIPT_ID: {SCRIPT_ID}, locals(): {locals()}")
    arg_config = locals()

    #### Parameter

    result_pkl = use_file(
        result_pkl.format(
            dataset_selected_slices_text=(str(dataset_selected_slices.replace(":", "-")))
        )
    )
    logger.debug(f"{result_pkl=}")
    huggingface_offline_cache_dir = Path(huggingface_offline_cache_dir)
    pt_ckpt = use_dir(pt_ckpt)
    dataset_selected_slices = slices.parse(dataset_selected_slices)

    #### [LSDist] Config

    model_config_dic = srsly.read_yaml(use_file(pt_ckpt / tag / "model_config.yml", new=False))
    logger.debug(f"{model_config_dic=}")
    pt_config_dic = srsly.read_json(use_file(pt_ckpt / tag / "pt_config.json", new=False))
    logger.debug(f"{pt_config_dic=}")

    #### [LLaMA] Config

    generation_config_dic: LlamaGenerationConfig = srsly.read_yaml(
        use_file(generation_config_yml, new=False)
    )
    logger.info(f"{generation_config_dic=}")
    temperature = (
        generation_config_dic["temperature"] if new_temperature is None else new_temperature
    )
    logger.info(f"{temperature=}")

    #### [LSDist] Model

    backbone: backbone_module.TBackbone = getattr(
        backbone_module, model_config_dic["backbone"]["class"]
    )(**model_config_dic["backbone"]["kwargs"])
    model = DiscreteDistributionNet(backbone, max_seq_len=generation_config_dic["max_seq_len"])
    model = model.to(DEVICE)
    logger.info(f"{model=!r}")

    model_state_fn = use_file(pt_ckpt / tag / "model_state.pt", new=False)
    model.load_state_dict(torch.load(model_state_fn, map_location=DEVICE))
    logger.info(f"Load {model_state_fn=}")

    #### [LLaMA] Dataset

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

    dataset = dataset.select(dataset_indices)
    assert len(dataset) == len(dataset_indices)

    logger.info(repr(repr(dataset)))
    logger.success(f"Init {dataset_name} dataset done!")

    #### [LLaMA] Model

    generator = Llama.build(
        ckpt_dir=llama_ckpt_dir,
        tokenizer_path=llama_tokenizer_model,
        max_seq_len=generation_config_dic["max_seq_len"],
        max_batch_size=dist_batch_size,
        enable_bfloat16=enable_bfloat16,
    )
    hidden_indices = slices.apply(range(generator.model.params.n_layers), (-1,))

    #### Dist & Generate loop

    dispatcher = LlamaDistDispatcher(generator, dist_model=model, dev=DEV)
    num_chunks = math.ceil(len(dataset) / chunk_size)

    result_dic = dict(arg_config=arg_config, every_chunk=[])
    for chunk_idx, example_chunk_and_indices in enumerate(
        chunked(zip(dataset, dataset_indices), chunk_size)
    ):
        example_chunk, example_indices = zip(*example_chunk_and_indices)
        logger.debug(f"Sample {chunk_idx=}/{num_chunks} start")

        dist_chunk, prompt_length_chunk = dispatcher.dist_from_chunk(
            example_chunk, dist_batch_size=dist_batch_size, hidden_indices=hidden_indices
        )
        result_dic["every_chunk"].append(
            dict(
                chunk_idx=chunk_idx,
                example_chunk=example_chunk,
                example_indices=example_indices,
                dist_chunk=dist_chunk,
                prompt_length_chunk=prompt_length_chunk,
            )
        )

    result_pkl.write_bytes(srsly.pickle_dumps(result_dic))
    logger.success(f"Save {result_pkl=}")


if __name__ == "__main__":
    fire.Fire(main)
