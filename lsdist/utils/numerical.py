import random
import torch
import numpy as np
import torch
from torch import Tensor
from typing import Any, Sequence
from numpy.typing import NDArray

from . import hash_obj

DEFAULT_ROOT_SEED = "Institute of Multimedia Knowledge Fusion and Engineering"


def init_random(root_seed: Any = DEFAULT_ROOT_SEED):
    """Initialize random seed of random, numpy and torch.
    `root_seed` can be any JSON-serializable object.
    """
    int_256bit_seed = hash_obj(root_seed, bit_length=256)
    int_32bit_seeds = ((int_256bit_seed >> (i * 32)) & ((1 << 32) - 1) for i in range(8))

    random.seed(next(int_32bit_seeds))
    np.random.seed(next(int_32bit_seeds))
    torch.manual_seed(next(int_32bit_seeds))
    torch.cuda.manual_seed(next(int_32bit_seeds))

    torch.backends.cudnn.deterministic = True


def custom_repr(value: Any) -> str:
    """
    Returns a custom string representation of the given value,
    with special handling for lists, numpy arrays, tensors, and floating-point numbers.
    """
    if isinstance(value, list):
        return f"list(length={len(value)})"
    elif isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    elif isinstance(value, torch.Tensor):
        return f"tensor(shape={value.size()}, dtype={value.dtype})"
    elif isinstance(value, float):
        return f"{value:.3g}"
    else:
        return repr(value)


def repr_for_dataclass(obj: Any) -> str:
    """
    The __repr__ method for a dataclass that uses the custom_repr function for its fields.
    """
    fields_str = [
        f"{field_name}={custom_repr(field_value)}"
        for field_name, field_value in obj.__dict__.items()
    ]
    return f"{obj.__class__.__name__}({', '.join(fields_str)})"


DOT5_LOG_2PI = 0.5 * np.log(2 * np.pi)


def gmm_nll(mu: Tensor, sigma: Tensor, log_p: Tensor, x: Tensor):
    r"""Negative Log-Likelihood of Gaussian Mixture Model

    $L(\theta; x) = -\log \left( \sum_{k=1}^{K} p_k \cdot \frac{1}{\sqrt{2\pi\sigma_k^2}}
    \exp \left( -\frac{(x - \mu_k)^2}{2\sigma_k^2} \right) \right)$

    - mu, sigma, log_p, x: [batch_size, num_samples, num_components]
    - mu, sigma, log_p, x: [batch_size, num_components]
    """
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    log_gaussians = exponent - torch.log(sigma) - DOT5_LOG_2PI
    log_probs = log_p + log_gaussians
    log_sum_exp = torch.logsumexp(log_probs, dim=-1)
    nll = -torch.mean(log_sum_exp)
    return nll


def gmm_pdf(mu: Tensor, sigma: Tensor, log_p: Tensor, x: Tensor):
    r"""Probability density function of Gaussian Mixture Model

    $p(x) = \sum_{k=1}^{K} p_k \mathcal{N}(x | \mu_k, \sigma_k^2)$

    $p(x) = \sum_{k=1}^{K} p_k \frac{1}{\sqrt{2\pi \sigma_k^2}}
    \exp \left(-\frac{(x - \mu_k)^2}{2\sigma_k^2}\right)$

    - mu, sigma, log_p, x: [batch_size, num_samples, num_components]
    - mu, sigma, log_p, x: [batch_size, num_components]
    """
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    log_gaussians = exponent - torch.log(sigma) - DOT5_LOG_2PI
    log_probs = log_p + log_gaussians
    log_sum_exp = torch.logsumexp(log_probs, dim=-1)
    pdf = torch.exp(log_sum_exp)
    return pdf


SQRT_2 = np.sqrt(2)


def gmm_cdf(mu: Tensor, sigma: Tensor, log_p: Tensor, x: Tensor):
    r"""Cumulative distribution function of Gaussian Mixture Model.
    https://www.johndcook.com/erf_and_normal_cdf.pdf

    $\Phi(x) = \sum_{k=1}^{K} p_k \Phi_{\mu_k, \sigma_k}(x)$
    $\Phi_{\mu_k, \sigma_k}(x) = \frac{1}{2} \left( 1 + \text{erf}\left( \frac{x - \mu_k}{\sqrt{2}\sigma_k} \right) \right)$
    $\Phi(x) = \frac{1}{2} \left( 1 + \sum_{k=1}^{K} p_k \text{erf}\left( \frac{x - \mu_k}{\sqrt{2}\sigma_k} \right) \right)$

    - mu, sigma, log_p, x: [batch_size, num_samples, num_components]
    - mu, sigma, log_p, x: [batch_size, num_components]
    """
    erfs = torch.erf((x - mu) / SQRT_2 / sigma)
    cdf = 0.5 * (1 + torch.sum(log_p.exp() * erfs, dim=-1))
    return cdf


def split_dataset(
    data: Sequence, proportions: tuple[float, ...], seed: int = None
) -> tuple[tuple, ...]:
    """Splits a dataset into multiple subsets based on given proportions.

    >>> split_dataset(range(10), (0.5, 0.3, 0.2), 42)
    ((5, 6, 0, 7, 3), (2, 4, 9), (1, 8))

    >>> data = range(100)
    >>> splits = split_dataset(data, (0.7, 0.15, 0.15), 42)
    >>> sorted(sum(splits, ())) == list(data)  # Test if recombined and sorted data is the same as original
    True
    """
    if sum(proportions) != 1:
        raise ValueError("The sum of the proportions must equal 1.")

    total_size = len(data)
    indexes = np.arange(total_size)
    np.random.default_rng(seed).shuffle(indexes)

    split_points = np.cumsum([int(round(p * total_size)) for p in proportions])
    split_indexes = np.split(indexes, split_points[:-1])
    return tuple(tuple(data[idx] for idx in sub_indexes) for sub_indexes in split_indexes)


def batched_bincount(input: Tensor, dim: int, num_value: int):
    """https://discuss.pytorch.org/t/batched-bincount/72819/4"""
    target_shape = (*input.shape[:dim], num_value + 1, *input.shape[dim + 1 :])
    target = torch.zeros(target_shape, dtype=input.dtype, device=input.device)

    index = torch.where(input < 0, torch.full_like(input, num_value), input)
    src = torch.ones_like(input)
    target.scatter_add_(dim, index, src)

    slices = [slice(None)] * len(target_shape)
    slices[dim] = slice(None, -1)
    return target[tuple(slices)]


def sorted_with_indices(lst, key=lambda x: x, reverse=False):
    """
    Sort a list and return the sorted elements along with their original indices.

    Parameters:
    lst: The list to be sorted.
    key: A function that extracts a comparison key from each list element.
    reverse: Whether to sort the elements in reverse order.

    Returns:
    Two lists: one containing the original indices of the sorted elements, and the other containing the sorted elements.

    Example:
    >>> prompt_chunk = ["apple", "banana", "cherry", "date"]
    >>> indices, sorted_chunk = sorted_with_indices(prompt_chunk, key=len)
    >>> indices
    [3, 0, 1, 2]
    >>> sorted_chunk
    ['date', 'apple', 'banana', 'cherry']
    """
    indexed_lst = sorted(enumerate(lst), key=lambda x: key(x[1]), reverse=reverse)
    indices, sorted_elements = zip(*indexed_lst)
    return list(indices), list(sorted_elements)


def generate_request_times(rate: float, num_requests: int, seed: int = None) -> NDArray[np.float64]:
    """
    Generate an array of request timestamps based on a Poisson process using a local random number generator.

    Parameters:
    rate: The expected number of requests per second (lambda in Poisson distribution).
    num_requests: The total number of requests to be generated.
    seed: Seed for the random number generator for reproducibility (optional).

    Returns:
    An array of timestamps (in seconds) when requests are made.

    Example:
    >>> times = generate_request_times(1, 5, 42)
    >>> np.allclose(times, [2.4042086, 4.74039826, 7.12515926, 7.40495355, 7.49139095])
    True
    """
    rng = np.random.default_rng(seed)
    inter_arrival_times = rng.exponential(scale=1 / rate, size=num_requests)
    request_times = np.cumsum(inter_arrival_times)
    return request_times


if __name__ == "__main__":
    import doctest

    doctest.testmod()
