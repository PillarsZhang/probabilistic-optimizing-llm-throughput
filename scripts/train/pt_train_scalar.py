import functools
from numbers import Number
from pathlib import Path
from typing import Callable
import fire
import srsly
import wandb
from loguru import logger
from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

import lsdist.model.backbone as backbone_module
from lsdist.model.net import ScalarNet
from lsdist.model.transforms import AddGaussianNoiseTransform
from lsdist.utils import get_env, get_script_id, get_ver, tic
from lsdist.utils.io import get_random_hex, use_dir, use_file
from lsdist.utils.log import init_loguru_logger
from lsdist.utils.numerical import init_random
from lsdist.dataset.trainable_lmdb import TrainableLMDBDataset, TrainableLMDBTransform

DEV = get_env() == "development"
SCRIPT_ID = get_script_id(__file__) + get_ver()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    comment: str = None,
    quantile: float = 0.8,
    num_epochs: int = 300,
    model_config_yml: str = "scripts/train/model_config.yml",
    pt_config_json: str = "scripts/train/pt_config.json",
    trainable_lmdb: str = "saved/sample/dist_to_trainable/trainable_lmdb/version=5,pooling=last",
    train_max_num_samples: int = None,
    train_max_num_items: int = None,
    prompt_only: bool = True,
    data_dropout: float = 0,
    data_noise: float = 1.0,
    pick_key: str = "valid/loss",
    pt_ckpt: str = f"saved/{SCRIPT_ID}/pt_ckpt",
    ckpt_jsonl: str = f"saved/{SCRIPT_ID}/ckpt.jsonl",
    global_log: str = f"saved/{SCRIPT_ID}/global_log/{{timestamp}}_{{log_id}}.log",
    no_wandb: bool = DEV,
    no_ckpt: bool = DEV,
):
    init_random()
    init_loguru_logger(use_file(global_log))
    logger.info(f"{SCRIPT_ID=}, {locals()=}")

    #### Parameter

    if comment is None:
        comment = get_random_hex(6)
    logger.info(f"{comment=}")

    #### Dataset & Data augmentation

    raw_dataset = TrainableLMDBDataset(trainable_lmdb)
    if train_max_num_samples is None:
        train_max_num_samples = raw_dataset.max_num_samples
    valid_max_num_samples = raw_dataset.max_num_samples

    train_dataset = raw_dataset.select(
        split_name="train", prompt_only=prompt_only, max_num_items=train_max_num_items
    )
    train_dataset.transform = TrainableLMDBTransform(
        train_max_num_samples, raw_dataset.max_seq_len, quantile=quantile
    )
    logger.info(f"{train_dataset=!r}")

    valid_dataset = raw_dataset.select(
        split_name="valid", prompt_only=prompt_only, max_num_items=None
    )
    valid_dataset.transform = TrainableLMDBTransform(
        valid_max_num_samples, raw_dataset.max_seq_len, quantile=quantile
    )
    logger.info(f"{valid_dataset=!r}")

    logger.success(f"Init dataset done!")

    data_augment_lst = []
    if data_dropout > 0:
        data_augment_lst.append(nn.Dropout(p=data_dropout))
    if data_noise > 0:
        data_augment_lst.append(AddGaussianNoiseTransform(std=data_noise))
    data_augment = nn.Sequential(*data_augment_lst)
    logger.info(f"{data_augment=!r}")

    #### Config

    model_config_dic = srsly.read_yaml(model_config_yml)
    logger.debug(f"{model_config_dic=}")

    pt_config_dic = srsly.read_json(pt_config_json)
    logger.debug(f"{pt_config_dic=}")

    #### Model

    backbone: backbone_module.TBackbone = getattr(
        backbone_module, model_config_dic["backbone"]["class"]
    )(**model_config_dic["backbone"]["kwargs"])
    model = ScalarNet(backbone, max_seq_len=raw_dataset.max_seq_len)
    model = model.to(DEVICE)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(f"{model=!r}")

    loss_fn = functools.partial(ScalarNet.loss_fn, max_seq_len=raw_dataset.max_seq_len)
    logger.success(f"Init model: net, loss_fn done!")

    #### Optimizer (optional LR Scheduler)

    optimizer: Optimizer = getattr(optim, pt_config_dic["optimizer"]["type"])(
        model_parameters, **pt_config_dic["optimizer"]["params"]
    )
    logger.info(f"{optimizer=!r}")

    if "lr_scheduler" in pt_config_dic:
        lr_scheduler: LRScheduler = getattr(
            optim.lr_scheduler, pt_config_dic["lr_scheduler"]["type"]
        )(optimizer, **pt_config_dic["lr_scheduler"]["params"])
    else:
        lr_scheduler = None
    logger.info(f"{lr_scheduler=!r}")
    logger.success(f"Init Optimizer (optional LR Scheduler) done!")

    #### Dataloader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=pt_config_dic["train_batch_size"],
        num_workers=pt_config_dic["train_num_workers"],
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=pt_config_dic["valid_batch_size"],
        num_workers=pt_config_dic["valid_num_workers"],
        shuffle=False,
    )
    logger.success(f"Init Dataloader done!")

    #### W&B and ckpt

    wandb.init(
        name=comment,
        project=f"lsdist-{SCRIPT_ID.replace('/', '-')}",
        mode="disabled" if no_wandb else "online",
        config={"model": model_config_dic, "pt": pt_config_dic},
    )
    logger.success(f"Init W&B done!")

    if not no_ckpt:
        ckpt_jsonl = use_file(ckpt_jsonl)
        pt_ckpt: Path = use_dir(pt_ckpt)
        logger.info(f"{ckpt_jsonl=}, {pt_ckpt=}")

    #### Loop

    global_steps = 0
    client_states = []
    for epoch in range(num_epochs):
        client_state = {"comment": comment, "epoch": epoch}
        epoch_toc = tic()

        # Train
        train_loss, num_steps = train_epoch(
            model, train_dataloader, loss_fn, optimizer, lr_scheduler, data_augment
        )
        client_state.update({"train/loss": train_loss})
        global_steps += num_steps
        client_state.update(global_steps=global_steps)
        last_lr = optimizer.param_groups[0]["lr"]
        client_state.update(last_lr=last_lr)

        # Valid
        valid_loss, _ = eval_epoch(model, valid_dataloader, loss_fn)
        client_state.update({"valid/loss": valid_loss})

        time_cost = epoch_toc()
        tag = f"{comment}>{epoch=},{train_loss=:g},{valid_loss=:g}"
        client_state.update(tag=tag, time_cost=time_cost)

        # Logging
        logger.info(
            f"Train | {comment} > "
            f"epoch: [{epoch}/{num_epochs}], {global_steps=}, {last_lr=:g} | "
            f"{train_loss=:g}, {valid_loss=:g} | "
            f"{time_cost=:.3f}s"
        )
        wandb.log({k: v for k, v in client_state.items() if isinstance(v, Number)})

        # Persistence
        if not no_ckpt:
            srsly.write_jsonl(ckpt_jsonl, [client_state], append=True, append_new_line=False)
            cond = (
                not pick_key
                or not client_states
                or (client_state[pick_key] < min(c[pick_key] for c in client_states))
            )
            if cond:
                srsly.write_yaml(use_file(pt_ckpt / tag / "model_config.yml"), model_config_dic)
                srsly.write_json(use_file(pt_ckpt / tag / "pt_config.json"), pt_config_dic)
                srsly.write_yaml(use_file(pt_ckpt / tag / "client_state.yml"), client_state)
                torch.save(model.state_dict(), use_file(pt_ckpt / tag / "model_state.pt"))
                # torch.save(optimizer.state_dict(), use_file(pt_ckpt / tag / "optimizer_state.pt"))

        client_states.append(client_state)

    if pick_key:
        best_pick_item = min(client_states, key=lambda c: c[pick_key])
        logger.info(
            f"{best_pick_item['comment']=}, {best_pick_item['epoch']=}, {best_pick_item[pick_key]=:g}"
        )
    logger.success(f"[{SCRIPT_ID}] done!")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    lr_scheduler: LRScheduler = None,
    data_augment: Callable[[Tensor], Tensor] = None,
):
    model.train()

    epoch_loss = 0.0
    epoch_batch = 0
    num_steps = 0

    dataloader_pbar = tqdm(dataloader, desc="train_epoch", dynamic_ncols=True, leave=False)
    for src, trg in dataloader_pbar:
        src: torch.Tensor = src.to(DEVICE)
        trg: torch.Tensor = trg.to(DEVICE)

        if data_augment:
            src = data_augment(src)

        optimizer.zero_grad()
        prd: Tensor = model(src)
        loss = loss_fn(prd, trg)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * prd.shape[0]
        epoch_batch += prd.shape[0]
        num_steps += 1

    if lr_scheduler is not None:
        lr_scheduler.step()

    avg_loss = epoch_loss / epoch_batch
    return avg_loss, num_steps


def eval_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: Callable):
    model.eval()

    epoch_loss = 0.0
    epoch_batch = 0
    num_steps = 0

    dataloader_pbar = tqdm(dataloader, desc="eval_epoch", dynamic_ncols=True, leave=False)
    for src, trg in dataloader_pbar:
        src: torch.Tensor = src.to(DEVICE)
        trg: torch.Tensor = trg.to(DEVICE)

        with torch.no_grad():
            prd = model(src)
            loss = loss_fn(prd, trg)

        epoch_loss += loss.item() * prd.shape[0]
        epoch_batch += prd.shape[0]
        num_steps += 1

    avg_loss = epoch_loss / epoch_batch
    return avg_loss, num_steps


if __name__ == "__main__":
    fire.Fire(main)
