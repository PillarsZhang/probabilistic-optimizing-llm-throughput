from typing import Any, Callable, Literal, TypedDict
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, PreTrainedModel
from loguru import logger
from pathlib import Path

from .io import use_dir


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


Dialog = list[Message]


class AlpacaExample(TypedDict):
    instruction: str
    input: str
    output: str
    text: str


def get_hf_cache_dir(name: str, home: str):
    cache_dir = Path(home).expanduser() / name.replace("/", "___")
    cache_exist = cache_dir.is_dir() and any(cache_dir.iterdir())
    return cache_dir, cache_exist


def use_hf_dataset(dataset_name: str, dataset_offline: bool, dataset_home: str):
    cache_dir, cache_exist = get_hf_cache_dir(dataset_name, dataset_home)
    full_name = f"dataset {dataset_name}"
    if dataset_offline:
        if cache_exist:
            logger.debug(f"Load {full_name} offline from {cache_dir} ...")
            dataset = load_from_disk(cache_dir)
        else:
            logger.debug(f"Cache does not exist, load {full_name} online ...")
            dataset = load_dataset(dataset_name)
            logger.debug(f"Save {full_name} to {cache_dir} ...")
            dataset.save_to_disk(use_dir(cache_dir))
    else:
        dataset = load_dataset(dataset_name)

    return dataset


def use_hf_model(
    model_name: str,
    model_offline: bool,
    model_home: str,
    from_pretrained_name: str = "model",
    from_pretrained_handle: Callable[
        [str, Any], PreTrainedModel
    ] = AutoModelForCausalLM.from_pretrained,
    **from_pretrained_kwargs,
):
    cache_dir, cache_exist = get_hf_cache_dir(model_name, model_home)
    full_name = f"{from_pretrained_name} {model_name}"
    if model_offline:
        if cache_exist:
            logger.debug(f"Load {full_name} offline from {cache_dir} ...")
            model = from_pretrained_handle(cache_dir, **from_pretrained_kwargs)
        else:
            logger.debug(f"Cache does not exist, load {full_name} online ...")
            model = from_pretrained_handle(model_name, **from_pretrained_kwargs)
            logger.debug(f"Save {full_name} to {cache_dir} ...")
            model.save_pretrained(use_dir(cache_dir))
    else:
        model = from_pretrained_handle(model_name, **from_pretrained_kwargs)

    return model
