from typing import List
import numpy as np
from numpy.typing import NDArray


def threshold(
    data: NDArray | List[float | int],
    thresholds: List[float] | float,
    bins: int = 10_001,
) -> List[float]:
    """
    Calculates Equal Error Rate (eer).
    Remember: Genuine scores provided must be greater in value compared to
    imposter.
    Can be used to calculate D-EER, by replacing imposter scores to morph
    scores.

    Parameters
    ----------------------------------------------------------------------
    data : List[float] | NDArray
        The list of data for which you want to calculate threshold.
    thresholds : List[float] | NDArray
        The list of thresholds in 0-1 scale.
    bins : int
        The number of bins to be considered while calculating thresholds.
        Default is 10_001.

    Returns
    ----------------------------------------------------------------------
    thresholds : float | List[float]
        Threshold score values for given thresholds.

    Example
    ----------------------------------------------------------------------
    Let's say you want to calculate thresholds at FMR == 0.1% and 0.01%.

    import biometrics

    imposter_scores = ... # imposter is a 1D numpy array or List of float
    thresholds = biometrics.threshold(
        imposter_scores,
        [1e-3, 1e-4], # supply the thresholds in 0-1 scale
        bins=10_001,
    )

    ----------------------------------------------------------------------

    """
    if not isinstance(thresholds, list) and not isinstance(thresholds, float):
        raise TypeError(
            f"Expected thresholds of type List[float] or float but got: {type(thresholds)}"
        )

    data = np.array(sorted(data, reverse=True))

    mi = np.min(data)
    mx = np.max(data)

    total_thresholds = np.linspace(mi, mx, bins)
    far = []
    for threshold in total_thresholds:
        fa = np.where(data >= threshold)[0].shape[0]
        far.append(fa / data.shape[0])

    far = np.array(far)

    if isinstance(thresholds, float):
        return total_thresholds[np.argmin(np.abs(far - thresholds))]

    if isinstance(thresholds, list):
        result: List[float] = []
        for threshold in thresholds:
            result.append(total_thresholds[np.argmin(np.abs(far - threshold))])

        return result
