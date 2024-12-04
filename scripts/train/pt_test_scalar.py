import functools
from typing import Callable
import fire
import srsly
from loguru import logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import lsdist.model.backbone as backbone_module
from lsdist.model.net import ScalarNet
from lsdist.utils import get_env, get_script_id, get_ver
from lsdist.utils.io import srsly_pickle_zlib_dumps, use_dir, use_file
from lsdist.utils.log import init_loguru_logger
from lsdist.utils.numerical import init_random
from lsdist.dataset.trainable_lmdb import TrainableLMDBDataset, TrainableLMDBTransform

DEV = get_env() == "development"
DEV_SUFFIX = "__DEV" if DEV else ""
SCRIPT_ID = get_script_id(__file__) + get_ver()
TRAIN_SCRIPT_ID = "train/pt_train_scalar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    quantile: float = 0.8,
    trainable_lmdb: str = "saved/sample/dist_to_trainable/trainable_lmdb/version=5,pooling=last",
    pick_comment: str = "20240528A-quantile-0.8-data-noise-0.8",
    pick_key: str = "valid/loss",
    prompt_only: bool = True,
    pt_ckpt: str = f"saved/{TRAIN_SCRIPT_ID}/pt_ckpt",
    ckpt_jsonl: str = f"saved/{TRAIN_SCRIPT_ID}/ckpt.jsonl",
    result_pkl: str = f"saved/{SCRIPT_ID}/result-quantile-{{quantile}}.pkl",
    global_log: str = f"saved/{SCRIPT_ID}/global_log/{{timestamp}}_{{log_id}}.log",
):
    init_random()
    init_loguru_logger(use_file(global_log))
    logger.info(f"SCRIPT_ID: {SCRIPT_ID}, locals(): {locals()}")

    #### Parameter

    pt_ckpt = use_dir(pt_ckpt)
    # assert f"-quantile-{quantile}-" in pick_comment

    #### Dataset

    raw_dataset = TrainableLMDBDataset(trainable_lmdb)
    test_max_num_samples = raw_dataset.max_num_samples

    test_dataset = raw_dataset.select(split_name="test", prompt_only=prompt_only)
    test_dataset.transform = TrainableLMDBTransform(
        test_max_num_samples, raw_dataset.max_seq_len, quantile=quantile
    )
    logger.info(f"{test_dataset=!r}")

    logger.success(f"Init dataset done!")

    #### Ckpt

    client_states = srsly.read_jsonl(ckpt_jsonl)
    client_states = list(client_states)
    if pick_comment is not None:
        client_states = list(filter(lambda c: c["comment"] == pick_comment, client_states))
    best_pick_item = min(client_states, key=lambda c: c[pick_key])
    logger.info(
        f"{best_pick_item['comment']=}, {best_pick_item['epoch']=}, {best_pick_item[pick_key]=:g}"
    )
    tag = best_pick_item["tag"]
    logger.info(f"Pick tag({repr(tag)}) for min({repr(pick_key)})")

    #### Config

    model_config_dic = srsly.read_yaml(use_file(pt_ckpt / tag / "model_config.yml", new=False))
    logger.debug(f"{model_config_dic=}")
    pt_config_dic = srsly.read_json(use_file(pt_ckpt / tag / "pt_config.json", new=False))
    logger.debug(f"{pt_config_dic=}")

    #### Model

    backbone: backbone_module.TBackbone = getattr(
        backbone_module, model_config_dic["backbone"]["class"]
    )(**model_config_dic["backbone"]["kwargs"])
    model = ScalarNet(backbone, max_seq_len=raw_dataset.max_seq_len)
    model = model.to(DEVICE)
    logger.info(f"{model=!r}")

    model_state_fn = use_file(pt_ckpt / tag / "model_state.pt", new=False)
    model.load_state_dict(torch.load(model_state_fn, map_location=DEVICE))
    logger.info(f"Load {model_state_fn=}")

    loss_fn = functools.partial(ScalarNet.loss_fn, max_seq_len=raw_dataset.max_seq_len)
    logger.success(f"Init model: net, loss_fn done!")

    #### Dataloader

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=pt_config_dic["valid_batch_size"],
        num_workers=pt_config_dic["valid_num_workers"],
        shuffle=False,
    )
    logger.success(f"Init Dataloader done!")

    #### Not Loop

    # Test
    test_loss, num_steps, trg, prd = eval_once(model, test_dataloader, loss_fn)
    logger.info(f"{test_loss=:g}, {num_steps=}")

    max_seq_len = raw_dataset.max_seq_len

    # Loss for each
    loss_each = []
    for _prd, _trg in zip(prd, trg):
        loss_each.append(loss_fn(_prd[None, :], _trg[None, :]))
    loss_each = torch.tensor(loss_each)  # [num_examples]
    assert torch.allclose(loss_each.sum() / prd.shape[0], torch.tensor(test_loss))

    # Find record
    _transform = test_dataset.transform
    test_dataset.transform = None
    record_lst = list(test_dataset)
    for r in record_lst:
        r.hidden = None
    test_dataset.transform = _transform

    result_dic = dict(
        prd=prd,
        trg=trg,
        test_loss=test_loss,
        loss_each=loss_each,
        record_lst=record_lst,
        max_seq_len=max_seq_len,
    )

    logger.info(f"result_dic: {result_dic.keys()}")

    with open(result_pkl := use_file(result_pkl.format(quantile=quantile)), "wb") as f:
        f.write(srsly_pickle_zlib_dumps(result_dic))
    logger.success(f"Save to {result_pkl}")


def eval_once(model: nn.Module, dataloader: DataLoader, loss_fn: Callable):
    model.eval()

    epoch_loss = 0.0
    epoch_batch = 0
    num_steps = 0

    trg_lst = []
    prd_lst = []

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

        trg_lst.append(trg.cpu())
        prd_lst.append(prd.cpu())

    avg_loss = epoch_loss / epoch_batch

    trg = torch.cat(trg_lst, dim=0)
    prd = torch.cat(prd_lst, dim=0)

    return avg_loss, num_steps, trg, prd


if __name__ == "__main__":
    fire.Fire(main)
