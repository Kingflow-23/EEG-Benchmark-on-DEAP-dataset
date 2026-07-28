"""
Dependency-free reimplementation of ``pyeeg.bin_power``.

The reference repo calls ``pyeeg.bin_power(x, band, sample_rate)`` once per
channel per window. ``pyeeg`` is unmaintained, is not installable from PyPI on
modern Python, and the function itself is ten lines, so we reimplement it here
and vectorise it over arbitrary leading dimensions.

The original (pyeeg 0.4.4) reads:

    def bin_power(X, Band, Fs):
        C = numpy.fft.fft(X)
        C = abs(C)
        Power = numpy.zeros(len(Band) - 1)
        for Freq_Index in range(0, len(Band) - 1):
            Freq      = float(Band[Freq_Index])
            Next_Freq = float(Band[Freq_Index + 1])
            Power[Freq_Index] = sum(
                C[int(numpy.floor(Freq      / Fs * len(X))):
                  int(numpy.floor(Next_Freq / Fs * len(X)))]
            )
        Power_Ratio = Power / sum(Power)
        return Power, Power_Ratio

Two properties worth stating plainly, because the name is misleading:

* It sums FFT **magnitudes**, not squared magnitudes. So this is not power in
  the physical sense (no Welch/periodogram normalisation, no window function,
  no scaling by sample rate or window length). It is a raw magnitude sum over
  a set of DFT bins. We keep it identical anyway -- deviating would silently
  change the features the published architecture was tuned on.
* Bin edges are computed with ``floor``, and the upper edge is exclusive, so a
  band [4, 8) at Fs=128 with a 256-sample window covers DFT bins 8..15.

The repo uses the first return value (absolute magnitude sums), not the ratio.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from ..config import BAND_EDGES, SAMPLE_RATE


def band_bin_indices(
    band_edges: Sequence[float],
    n_samples: int,
    sample_rate: float = SAMPLE_RATE,
) -> list[Tuple[int, int]]:
    """Map frequency-band edges to half-open DFT-bin ranges.

    Parameters
    ----------
    band_edges
        Increasing frequency edges in hertz. Adjacent values define one band.
    n_samples
        Signal length used by the DFT.
    sample_rate
        Sampling frequency in hertz.

    Returns
    -------
    list of tuple of int
        One ``(low, high)`` half-open bin range per band.

    Raises
    ------
    ValueError
        If any band is too narrow to contain a bin at this resolution.
    """
    idx = []
    for lo_hz, hi_hz in zip(band_edges[:-1], band_edges[1:]):
        lo = int(np.floor(float(lo_hz) / sample_rate * n_samples))
        hi = int(np.floor(float(hi_hz) / sample_rate * n_samples))
        if hi <= lo:
            raise ValueError(
                f"Band {lo_hz}-{hi_hz} Hz maps to an empty bin range [{lo}, {hi}) "
                f"at n_samples={n_samples}, sample_rate={sample_rate}."
            )
        idx.append((lo, hi))
    return idx


def bin_power(
    x: np.ndarray,
    band_edges: Sequence[float] = BAND_EDGES,
    sample_rate: float = SAMPLE_RATE,
    return_ratio: bool = False,
) -> np.ndarray:
    """Vectorised ``pyeeg.bin_power``.

    Parameters
    ----------
    x
        Signal array of shape ``(..., n_samples)``. The FFT is taken over the
        last axis; all leading axes are treated as independent signals, so this
        handles ``(n_channels, n_windows, window_size)`` in a single call.
    band_edges
        ``len(bands) + 1`` frequency edges in Hz, e.g. ``[4, 8, 12, 16, 25, 45]``.
    sample_rate
        Hz.
    return_ratio
        If True, return each band normalised by the total across bands
        (pyeeg's second return value) instead of the absolute sums.

    Returns
    -------
    numpy.ndarray
        ``float64`` values shaped ``(..., len(band_edges) - 1)``. Values are
        magnitude sums unless ``return_ratio`` is true.

    Raises
    ------
    ValueError
        If a band is empty or extends beyond the non-negative Nyquist range.
    """
    x = np.asarray(x)
    n = x.shape[-1]
    bins = band_bin_indices(band_edges, n, sample_rate)

    # pyeeg uses the full complex FFT. Every bin index we need sits at or below
    # the Nyquist bin, where rfft is bit-for-bit identical and half the work.
    highest_bin = max(hi for _, hi in bins)
    if highest_bin > n // 2 + 1:
        raise ValueError(
            f"Band edges reach DFT bin {highest_bin}, above the Nyquist bin "
            f"{n // 2} for a {n}-sample window at {sample_rate} Hz. "
            "Lower the top band edge or lengthen the window."
        )

    mag = np.abs(np.fft.rfft(x, axis=-1))

    out = np.empty(x.shape[:-1] + (len(bins),), dtype=np.float64)
    for i, (lo, hi) in enumerate(bins):
        out[..., i] = mag[..., lo:hi].sum(axis=-1)

    if return_ratio:
        total = out.sum(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(total > 0, out / total, 0.0)

    return out
