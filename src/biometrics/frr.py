from typing import List
import numpy as np
from numpy.typing import NDArray


def frr(
    genuine: NDArray | List[float | int],
    thresholds: float | List[float],
) -> float | List[float]:
    """
    Calculates False Reject Rate (frr).
    Remember: Genuine scores provided must be greater in value compared to
    imposter.

    Parameters
    ----------------------------------------------------------------------
    genuine : List[float] | NDArray
        The list of genuine scores.

    Returns
    ----------------------------------------------------------------------
    frr : float | List[float]
        False Reject Rate (eer) calculated from given genuine scores.

    Example
    ----------------------------------------------------------------------
    Let's say you want to calculate FRR at FMR == 0.1% and 0.01%.

    import biometrics

    genuine_scores = ... # genuine is a 1D numpy array or List of float
    thresholds = biometrics.threshold(
        imposter_scores,
        [1e-3, 1e-4], # supply the thresholds in 0-1 scale
        bins=10_001,
    )
    frrs = biometrics.frr(genuine_scores, thresholds)
    print('FRR at FMR = 0.1%:', frrs[0])
    print('FRR at FMR = 0.01%:', frrs[1])

    ----------------------------------------------------------------------

    """
    if not isinstance(thresholds, list) and not isinstance(thresholds, float):
        raise TypeError(
            f"Expected thresholds of type List[float] or float but got: {type(thresholds)}"
        )

    genuine = np.squeeze(np.array(genuine))
    if isinstance(thresholds, float):
        return len(np.where(genuine <= thresholds)[0]) / genuine.shape[0]
    if isinstance(thresholds, list):
        resutls: List[float] = []
        for thres in thresholds:
            resutls.append(len(np.where(genuine <= thres)[0]) / genuine.shape[0])
        return resutls
