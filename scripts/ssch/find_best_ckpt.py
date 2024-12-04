from itertools import groupby
from operator import itemgetter
import fire
import srsly
from loguru import logger

from lsdist.utils import get_script_id, get_ver
from lsdist.utils.log import init_loguru_logger
from lsdist.utils.numerical import init_random

SCRIPT_ID = get_script_id(__file__) + get_ver()
TRAIN_SCRIPT_ID = "ssch/pt_train"


def main(
    pick_key: str = "valid/loss",
    ckpt_jsonl: str = f"saved/{TRAIN_SCRIPT_ID}/ckpt.jsonl",
):
    init_random()
    init_loguru_logger()
    logger.info(f"SCRIPT_ID: {SCRIPT_ID}, locals(): {locals()}")

    #### Ckpt
    client_states: list[dict] = srsly.read_jsonl(ckpt_jsonl)
    client_states = sorted(client_states, key=itemgetter("comment"))

    logger.info("All")
    find(client_states, pick_key)

    for comment, group in groupby(client_states, key=itemgetter("comment")):
        logger.info(f"comment={comment}")
        find(group, pick_key)


def find(client_states: list[dict], pick_key: str):
    best_pick_item = min(client_states, key=lambda c: c[pick_key])
    logger.debug(best_pick_item)
    logger.info(
        f"best_pick_item | comment: {best_pick_item['comment']}, "
        f"epoch: {best_pick_item['epoch']}, {pick_key}: {best_pick_item[pick_key]:.5f}"
    )


if __name__ == "__main__":
    fire.Fire(main)
