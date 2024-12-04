# https://github.com/zhengzangw/Sequence-Scheduling/blob/main/src/utils.py
# @article{zheng2023response,
#     title={Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline},
#     author={Zangwei Zheng and Xiaozhe Ren and Fuzhao Xue and Yang Luo and Xin Jiang and Yang You},
#     journal={arXiv preprint arXiv:2305.13144},
#     year={2023}
# }

from collections import defaultdict
import numpy as np


def buckit(x, cell: int = 50):
    x = int(np.ceil(x / cell) * cell)
    return x


# ===
# Scheduler
# ===

len_bs_dict = {
    200: 256,
    400: 128,
    600: 64,
    800: 32,
    10000: 16,
}

l_arr = np.array(list(len_bs_dict.keys()), dtype=int)
bs_arr = np.array(list(len_bs_dict.values()), dtype=int)


def len_bs_dict_fn(x):
    y = l_arr - x
    y[y < 0] = 10000
    bs = bs_arr[np.argmin(y)]
    return bs


def schedule(
    lengths: list[int], mini_batch_size: int = 1, vbs: bool = False, cell: int = 50
) -> list[tuple[list[int], int]]:
    # sort ids by length
    lengths_with_id = [(i, l) for i, l in enumerate(lengths)]
    sorted_lengths_with_id = sorted(lengths_with_id, key=lambda x: x[1], reverse=False)

    # batchify
    batches = []
    if not vbs:
        for i in range(0, len(lengths), mini_batch_size):
            batch = sorted_lengths_with_id[i : i + mini_batch_size]
            batch_ids = [x[0] for x in batch]
            max_len = max([x[1] for x in batch])
            batches.append((batch_ids, max_len))
    else:
        # group by length
        ids_len_dict = defaultdict(list)
        for i, l in sorted_lengths_with_id:
            ids_len_dict[buckit(l, cell)].append(i)
        # batchify
        max_l = max(lengths)
        for l, ids in ids_len_dict.items():
            bs = len_bs_dict_fn(l)
            for i in range(0, len(ids), bs):
                batch = ids[i : i + bs]
                if l < max_l and len(batch) < max(len_bs_dict_fn(l) // 2, 16):
                    l_ = l + cell
                    while l_ not in ids_len_dict:
                        l_ += cell
                    ids_len_dict[l_].extend(batch)
                    break
                batches.append((batch, l))
    return batches
