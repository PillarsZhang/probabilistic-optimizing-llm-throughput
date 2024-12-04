import numpy as np
from numpy.typing import NDArray
from functools import reduce
from scipy.signal import convolve


def cdf2pdf(cdf: NDArray, axis: int = -1):
    return np.diff(cdf, prepend=0, axis=axis)


def pdf2cdf(pdf: NDArray, axis: int = -1):
    return np.cumsum(pdf, axis=axis)


def translate_response(pdfs: NDArray, prompt_lengths: list[int]):
    response_pdfs = np.zeros_like(pdfs)
    max_length = pdfs.shape[1] - 1
    for seq_idx, prompt_length in enumerate(prompt_lengths):
        response_pdfs[seq_idx, 1 : max_length - prompt_length + 1] = pdfs[
            seq_idx, prompt_length + 1 :
        ]
        response_pdfs[seq_idx, 0] = 1 - np.sum(
            response_pdfs[seq_idx, 1 : max_length - prompt_length + 1]
        )
    return response_pdfs


def sim_vbs(pdfs: NDArray, batches: list[tuple]):
    max_length = pdfs.shape[1] - 1
    cdfs = pdf2cdf(pdfs)
    batches_pdf = np.zeros((len(batches), max_length + 1), dtype=float)

    for batch_idx, batch in enumerate(batches):
        batches_pdf[batch_idx, :] = cdf2pdf(np.prod(cdfs[batch, :], axis=0))

    chunk_pdf = reduce(lambda x, y: convolve(x, y), batches_pdf)
    chunk_mean = np.sum(chunk_pdf * np.arange(chunk_pdf.shape[0]))
    return chunk_pdf, chunk_mean


def sim_fcr(pdfs: NDArray, batches: list[tuple], stops: list[int], fcr_batch_size: int):
    max_length = pdfs.shape[1] - 1
    cdfs = pdf2cdf(pdfs)
    batches_pdf = np.zeros((len(batches), max_length + 1), dtype=float)
    post_pdf_lst = []

    for batch_idx, batch in enumerate(batches):
        this_pdfs = pdfs[batch, :]

        prev_pdfs = np.zeros_like(this_pdfs)
        prev_pdfs[:, : stops[batch_idx]] = this_pdfs[:, : stops[batch_idx]]
        prev_pdfs[:, stops[batch_idx] + 1] = 1 - np.sum(prev_pdfs, axis=1)

        post_pdfs = np.zeros_like(this_pdfs)
        post_pdfs[:, 1 : max_length - stops[batch_idx] + 1] = this_pdfs[:, stops[batch_idx] + 1 :]
        post_pdfs[:, 0] = 1 - np.sum(post_pdfs, axis=1)

        # assert np.allclose(np.sum(prev_pdfs, axis=1), 1)
        # assert np.allclose(np.sum(post_pdfs, axis=1), 1)
        # assert np.allclose(
        #     np.sum(post_pdfs[:, : stops[batch_idx]], axis=1)
        #     + np.sum(post_pdfs[:, 1 : max_length - stops[batch_idx] + 1], axis=1),
        #     1,
        # )

        batches_pdf[batch_idx, :] = cdf2pdf(np.prod(pdf2cdf(prev_pdfs), axis=0))
        post_pdf_lst.extend(post_pdfs)

    chunk_pdf = reduce(lambda x, y: convolve(x, y), batches_pdf)
    # fcr_pdf = reduce(lambda x, y: convolve(x, y), post_pdf_lst)

    post_pdfs = np.array(post_pdf_lst, dtype=float)
    fcr_pdf = cdf2pdf(np.prod(pdf2cdf(post_pdfs), axis=0))
    fcr_mean = np.sum(fcr_pdf * np.arange(fcr_pdf.shape[0]))
    post_num_seq = np.sum(1 - post_pdfs[:, 0])
    fcr_num_batch = post_num_seq / fcr_batch_size
    fcr_mean = fcr_mean * fcr_num_batch

    chunk_mean = np.sum(chunk_pdf * np.arange(chunk_pdf.shape[0]))
    return chunk_pdf, chunk_mean, fcr_num_batch, fcr_mean
