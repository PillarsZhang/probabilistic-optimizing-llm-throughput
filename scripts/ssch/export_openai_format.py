import copy
import json
from pathlib import Path
import shutil
import fire
import jinja2
from loguru import logger
import numpy as np
from tqdm import tqdm
import srsly

import torch
from lsdist.llama.tokenizer import Tokenizer
from lsdist.utils import get_env, get_script_id, get_ver, slices
from lsdist.utils.data import use_hf_dataset
from lsdist.utils.numerical import init_random
from lsdist.dataset.trainable_lmdb import TrainableLMDBDataset

DEV = get_env() == "development"
SCRIPT_ID = get_script_id(__file__) + get_ver()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHUFFLE_SEED = 20231005


def main(
    trainable_lmdb: str = "saved/sample/dist_to_trainable/trainable_lmdb/version=5,pooling=last",
    llama_tokenizer_model: str = "../llama/tokenizer.model",
    scalar_rule: tuple = ("max",),
    ssch_prompt_template: str = f"scripts/ssch/ssch_prompt_template_{{scalar_fn}}.jinja",
    add_bin: int = 100,
    dataset_selected_slices_for_bench: str = "10000:",
    dataset_name: str = "tatsu-lab/alpaca",
    dataset_offline: bool = True,
    huggingface_offline_cache_dir: str = "~/.cache/huggingface_offline_cache",
    dataset_info_json: str = f"saved/{SCRIPT_ID}/dataset_info.json",
    export_fn: str = f"saved/{SCRIPT_ID}/my_{{scalar_name}}_{{split_name}}.json",
):
    init_random()
    logger.info(f"{SCRIPT_ID=}, {locals()=}")

    #### Parameter

    huggingface_offline_cache_dir = Path(huggingface_offline_cache_dir)
    dataset_selected_slices_for_bench = slices.parse(dataset_selected_slices_for_bench)
    dataset_info_json = Path(dataset_info_json)

    #### Prompt Dataset

    if dataset_name == "tatsu-lab/alpaca":
        dataset = use_hf_dataset(
            dataset_name, dataset_offline, huggingface_offline_cache_dir / "dataset"
        )
        dataset = dataset["train"].shuffle(SHUFFLE_SEED)
    else:
        raise f"Unknown dataset: {dataset_name}"

    bench_dataset_indices = slices.apply(range(len(dataset)), dataset_selected_slices_for_bench)
    logger.info(
        f"Filter dataset_selected_slices_for_bench {slices.format(dataset_selected_slices_for_bench)} -> {len(bench_dataset_indices)} examples"
    )

    logger.info(repr(repr(dataset)))
    logger.success(f"Init {dataset_name} dataset done!")
    logger.info(f"{len(bench_dataset_indices)=}")
    prompt_dataset = dataset

    tokenizer = Tokenizer(model_path=llama_tokenizer_model)

    #### Dist Dataset

    raw_dataset = TrainableLMDBDataset(trainable_lmdb)

    train_dataset = raw_dataset.select(split_name="train", prompt_only=True, max_num_items=None)
    logger.info(f"{train_dataset=!r}")

    valid_dataset = raw_dataset.select(split_name="valid", prompt_only=True, max_num_items=None)
    logger.info(f"{valid_dataset=!r}")

    test_dataset = raw_dataset.select(split_name="test", prompt_only=True, max_num_items=None)
    logger.info(f"{test_dataset=!r}")

    dist_dataset = dict(train=train_dataset, valid=valid_dataset, test=test_dataset)
    logger.success(f"Init dataset done!")

    #### Export

    scalar_fn, *scalar_args = scalar_rule
    # scalar_func return int
    if scalar_fn == "max":
        scalar_func = max
        scalar_name = f"{scalar_fn}"
    elif scalar_fn == "mean":
        scalar_func = lambda x: int(np.mean(x))
        scalar_name = f"{scalar_fn}"
    elif scalar_fn == "percentile":
        scalar_func = lambda x: int(np.percentile(x, scalar_args[0]))
        scalar_name = f"{scalar_fn}{scalar_args[0]}"
    elif scalar_fn == "minmax":
        scalar_func = lambda x: (min(x), max(x))
        scalar_name = f"{scalar_fn}"
    else:
        raise ValueError(f"Unknown scalar_fn: {scalar_fn}")

    if add_bin is not None:
        scalar_name = f"{scalar_name}_bin{add_bin}"

    llama_factory_dataset_info = {}
    if dataset_info_json.exists():
        llama_factory_dataset_info = srsly.read_json(dataset_info_json)

    _export_fn = export_fn

    # Load ssch prompt template (jinja2)
    ssch_prompt_template_fn = Path(ssch_prompt_template.format(scalar_fn=scalar_fn))
    logger.info(f"Load ssch prompt template from file: {ssch_prompt_template_fn}")
    ssch_prompt_template = jinja2.Template(ssch_prompt_template_fn.read_text())

    def format_oai(prompt_example: dict, dataset_idx: int, example: dict = None):
        if prompt_example["input"] == "":
            prompt = prompt_example["instruction"]
        else:
            prompt = prompt_example["instruction"] + "\n" + prompt_example["input"]

        _prompt = prompt
        prompt = ssch_prompt_template.render(input=prompt)
        if example:
            assert len(tokenizer.encode(_prompt, bos=True, eos=False)) == len(example.tokens) - 7
            num_response = scalar_func(example.post_lengths)
            num_response_before_add_bin = copy.copy(num_response)
            if add_bin is not None:
                # In our approach, we use bins with a cell size of 50 and round the numbers to the nearest bin that is greater than the actual length.
                if isinstance(num_response, (int, float)):
                    num_response = int(np.ceil(num_response / add_bin) * add_bin)
                elif isinstance(num_response, (list, tuple)) and len(num_response) == 2:
                    num_response = [
                        int(np.floor(num_response[0] / add_bin) * add_bin),
                        int(np.ceil(num_response[1] / add_bin) * add_bin),
                    ]
                else:
                    raise ValueError(f"Unknown num_response: {num_response}")

            if isinstance(num_response, (int, float)):
                response = f"{num_response}"
            elif isinstance(num_response, (list, tuple)) and len(num_response) == 2:
                response = f"{num_response[0]} {num_response[1]}"

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            return dict(
                messages=messages,
                dataset_idx=dataset_idx,
                num_response=num_response,
                num_response_before_add_bin=num_response_before_add_bin,
            )
        else:
            return dict(messages=[{"role": "user", "content": prompt}], dataset_idx=dataset_idx)

    def save_and_update(export_fn: Path, export_lst: list):
        export_fn.parent.mkdir(parents=True, exist_ok=True)
        srsly.write_json(export_fn, export_lst)
        logger.success(f"Exported {export_fn}")

        llama_factory_dataset_info[export_fn.stem] = {
            "file_name": export_fn.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }

        dataset_info_json.write_text(json.dumps(llama_factory_dataset_info, indent=4))
        logger.success(f"Updated {dataset_info_json}")

    for split_name, dataset in dist_dataset.items():
        export_fn: Path = Path(_export_fn.format(scalar_name=scalar_name, split_name=split_name))
        logger.info(f"Exporting {export_fn}")

        export_lst = []
        for example in tqdm(dataset, desc=f"Exporting {split_name}"):
            prompt_example = prompt_dataset[example.dataset_idx]
            export_lst.append(format_oai(prompt_example, example.dataset_idx, example))
        save_and_update(export_fn, export_lst)

    export_fn: Path = Path(_export_fn.format(scalar_name=scalar_name, split_name="bench"))
    logger.info(f"Exporting {export_fn}")
    export_lst = []
    for dataset_idx in tqdm(bench_dataset_indices, desc=f"Exporting bench"):
        prompt_example = prompt_dataset[dataset_idx]
        export_lst.append(format_oai(prompt_example, dataset_idx))
    save_and_update(export_fn, export_lst)


if __name__ == "__main__":
    fire.Fire(main)
