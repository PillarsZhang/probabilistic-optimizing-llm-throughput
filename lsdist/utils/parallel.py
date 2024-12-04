import os
import signal
import sys
import functools
import threading
import time
import psutil
import traceback
import multiprocessing as mp
import functools
from typing import Any, Callable
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor

MP_CONTEXT = mp.get_context("spawn")


def async_executor(
    executor: ThreadPoolExecutor | ProcessPoolExecutor,
    on_success: Callable[[Any], None] = None,
    on_error: Callable[[Exception], None] = None,
    callback_input: bool = False,
):
    """A utility that wraps a function to be submitted to an Executor
    for asynchronous execution."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            future = executor.submit(func, *args, **kwargs)

            def callback(future: Future):
                try:
                    result = future.result()
                    if on_success:
                        if not callback_input:
                            on_success(result)
                        else:
                            on_success(result, func, *args, **kwargs)
                except Exception as error:
                    if on_error:
                        if not callback_input:
                            on_error(error)
                        else:
                            on_error(error, func, *args, **kwargs)
                    else:
                        traceback.print_exception(error)
                        print(
                            f"Exception raised, pid-{os.getpid()} immediately interrupt",
                            file=sys.stderr,
                        )
                        kill_process_tree()

            future.add_done_callback(callback)
            return future

        return wrapper

    return decorator


def watch_parent_kill(interval=0.05):
    pid, ppid = os.getpid(), os.getppid()

    def f():
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                print(f"ppid-{ppid} died, pid-{pid} immediately interrupt", file=sys.stderr)
                os.kill(pid, signal.SIGTERM)
                break
            time.sleep(interval)

    threading.Thread(target=f, daemon=True).start()


def kill_process_tree(recursive=True):
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=recursive):
        print(f"kill_process_tree: child (ppid-{child.ppid}, pid-{child.pid})", file=sys.stderr)
        child.terminate()
    print(f"kill_process_tree: parent (ppid-{child.ppid}, pid-{child.pid})", file=sys.stderr)
    parent.terminate()


def simple_cache(shared=False, is_method=False):
    """Shared dictionary for multiprocessing or Regular dictionary for single-process use
    Warning: The lock mechanism and serialization mechanism are unclear
    """
    cache = mp.Manager().dict() if shared else dict()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_method:
                key = (args[1:], tuple(sorted(kwargs.items())))
            else:
                key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        return wrapper

    return decorator
