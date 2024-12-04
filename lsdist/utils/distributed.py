import os


def test_rank():
    return os.environ.get("RANK") is not None


def assert_rank():
    assert test_rank(), (
        "Unable to read the rank value from environment variable. "
        "Please use torch.distributed.launch or torchrun to launch the program."
    )


def get_rank_id():
    # https://pytorch.org/docs/stable/elastic/run.html#environment-variables
    assert_rank()
    global_rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    node_rank = os.environ.get("GROUP_RANK", "0")
    return f"{global_rank}>{node_rank}:{local_rank}"


def on_rank(global_rank: int = 0):
    assert_rank()
    current_global_rank = int(os.environ.get("RANK"))
    return current_global_rank == global_rank
