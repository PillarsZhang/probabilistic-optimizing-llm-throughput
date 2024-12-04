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
from lsdist.model.net import DiscreteDistributionNet, ScalarNet
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
    gen_batch_size: int = 32,
    batched_rule: tuple = (),
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
    result_pkl: str = f"saved/{SCRIPT_ID}/{{ssch_lengths_prefix}}{{dataset_selected_slices_text}}-{{batched_rule_text}}{{temperature_suffix}}{{model_scalar_suffix}}_result.pkl",
    global_log: str = f"saved/{SCRIPT_ID}/global_log/{{timestamp}}_{{log_id}}.log",
    ssch_lengths_name: str = None,
    ssch_lengths_key: str = "lora",
    new_temperature: float = None,
    model_scalar: bool = False,
    model_scalar_tag: str = None,
):
    init_random()
    init_loguru_logger(use_file(global_log))
    logger.info(f"SCRIPT_ID: {SCRIPT_ID}, locals(): {locals()}")
    arg_config = locals()

    #### Parameter

    result_pkl = use_file(
        result_pkl.format(
            ssch_lengths_prefix=(
                f"{ssch_lengths_name}_{ssch_lengths_key}-" if ssch_lengths_name is not None else ""
            ),
            batched_rule_text=("_".join(str(x) for x in batched_rule)),
            dataset_selected_slices_text=(str(dataset_selected_slices.replace(":", "-"))),
            temperature_suffix=(f"_T{new_temperature}" if new_temperature is not None else ""),
            model_scalar_suffix=(f"_scalar-{model_scalar_tag}" if model_scalar else ""),
        )
    )
    logger.debug(f"{result_pkl=}")
    huggingface_offline_cache_dir = Path(huggingface_offline_cache_dir)
    pt_ckpt = use_dir(pt_ckpt)
    dataset_selected_slices = slices.parse(dataset_selected_slices)
    ssch_lengths_json = (
        use_file(f"saved/ssch/get_ssch_predict/my_{ssch_lengths_name}_bench.json", new=False)
        if ssch_lengths_name
        else None
    )

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
    if model_scalar:
        model = ScalarNet(backbone, max_seq_len=generation_config_dic["max_seq_len"])
    else:
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

    #### Inject ssch lengths
    use_known_lengths = False
    if ssch_lengths_json is not None:
        use_known_lengths = True
        ssch_lengths_lst = srsly.read_json(ssch_lengths_json)
        len_key = f"{ssch_lengths_key}_num_response_predict"
        tc_key = f"{ssch_lengths_key}_response_time"
        logger.debug(f"{len_key=}, {tc_key=}")
        known_lengths_idx2len = {
            x["dataset_idx"]: x[len_key] for x in ssch_lengths_lst if len_key in x
        }
        known_lengths_idx2tc = {
            x["dataset_idx"]: x[tc_key] for x in ssch_lengths_lst if len_key in x
        }
        # Check if cover all dataset_indices
        assert set(known_lengths_idx2len.keys()).issuperset(set(dataset_indices))
        assert set(known_lengths_idx2tc.keys()).issuperset(dataset_indices)
        logger.success(f"Inject ssch_lengths done!")

    #### Dist & Generate loop

    dispatcher = LlamaDistDispatcher(generator, dist_model=model, dev=DEV)

    toc = tic()
    num_chunks = math.ceil(len(dataset) / chunk_size)
    generate_time_cost = 0.0

    result_dic = dict(arg_config=arg_config, every_chunk=[])
    for chunk_idx, example_chunk_and_indices in enumerate(
        chunked(zip(dataset, dataset_indices), chunk_size)
    ):
        example_chunk, example_indices = zip(*example_chunk_and_indices)
        logger.debug(f"Sample {chunk_idx=}/{num_chunks} start")

        known_lengths = None
        known_lengths_tc = None
        if use_known_lengths:
            known_lengths = [known_lengths_idx2len[x] for x in example_indices]
            known_lengths_tc = max([known_lengths_idx2tc[x] for x in example_indices])
            logger.debug(f"{len(known_lengths)=}, {known_lengths_tc=}")

        toc_chunk = tic()

        total_generation_num_tokens = 0
        total_generation_num_lengths = 0
        (
            generation_tokens,
            generation_strings,
            time_history,
            generation_num_tokens,
            generation_num_lengths,
            other_info,
        ) = dispatcher.sample_from_chunk(
            example_chunk,
            dist_batch_size=dist_batch_size,
            gen_batch_size=gen_batch_size,
            temperature=temperature,
            top_p=generation_config_dic["top_p"],
            max_gen_len=generation_config_dic["max_gen_len"],
            hidden_indices=hidden_indices,
            batched_rule=batched_rule,
            known_lengths=known_lengths,
            known_lengths_tc=known_lengths_tc,
            model_scalar=model_scalar,
        )
        result_dic["every_chunk"].append(
            dict(
                chunk_idx=chunk_idx,
                example_chunk=example_chunk,
                generation_tokens=generation_tokens,
                generation_strings=generation_strings,
                time_history=time_history,
                generation_num_tokens=generation_num_tokens,
                generation_num_lengths=generation_num_lengths,
                other_info=other_info,
            )
        )
        total_generation_num_tokens += generation_num_tokens
        total_generation_num_lengths += generation_num_lengths
        # logger.debug(f"{generation_strings=}")
        logger.debug(f"Sample {chunk_idx=} done, {toc_chunk()=:.3f}s, {time_history=}")
        generate_time_cost += time_history["generate"]
        if "re-generate" in time_history:
            generate_time_cost += time_history["re-generate"]
    time_cost = toc()
    if known_lengths_tc is not None:
        time_cost += known_lengths_tc
    throughput = len(dataset) / time_cost
    generate_throughput = len(dataset) / generate_time_cost
    generate_throughput_tokens = total_generation_num_tokens / generate_time_cost
    generate_throughput_lengths = total_generation_num_lengths / generate_time_cost
    effective_token_ratio = total_generation_num_tokens / total_generation_num_lengths
    logger.info(f"{time_cost=:.3f}s, {generate_time_cost=:.3f}s")
    logger.info(
        f"{batched_rule=}, {throughput=:.3f} samples/s, {generate_throughput=:.3f} samples/s"
    )
    logger.info(
        f"{generate_throughput_tokens=:.3f} tokens/s, {generate_throughput_lengths=:.3f} tokens/s"
    )
    logger.info(f"{effective_token_ratio=:.3%}")
    result_dic["summary"] = dict(
        time_cost=time_cost,
        throughput=throughput,
        generate_time_cost=generate_time_cost,
        generate_throughput=generate_throughput,
        generate_throughput_tokens=generate_throughput_tokens,
        generate_throughput_lengths=generate_throughput_lengths,
        total_generation_num_tokens=total_generation_num_tokens,
        total_generation_num_lengths=total_generation_num_lengths,
    )
    result_pkl.write_bytes(srsly.pickle_dumps(result_dic))
    logger.success(f"Save {result_pkl=}")


if __name__ == "__main__":
    fire.Fire(main)
