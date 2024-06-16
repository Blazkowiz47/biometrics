from typing import List
import numpy as np
from numpy.typing import NDArray


def iapar(
    morph: NDArray | List[float | int],
    thresholds: float | List[float],
) -> float | List[float]:
    """
    Calculates Impostor Attack Presentation Acceptance Rate (iapar).

    Parameters
    ----------------------------------------------------------------------
    genuine : List[float] | NDArray
        The list of genuine scores.

    Returns
    ----------------------------------------------------------------------
    eer : float
        Equal Error Rate (eer) calculated from given genuine and imposter
        scores

    Example
    ----------------------------------------------------------------------
    Let's say you want to calculate FRR at FMR == 0.1% and 0.01%.

    import biometrics

    impostor_scores = ... # impostor is a 1D numpy array or List of float
    morph_scores = ... # morph is a 1D numpy array or List of float
    thresholds = biometrics.threshold(
        imposter_scores,
        [1e-3, 1e-4], # supply the thresholds in 0-1 scale
        bins=10_001,
    )
    iapars = biometrics.frr(morph_scores, thresholds)
    print('IAPAR at FMR = 0.1%:', iapars[0])
    print('IAPAR at FMR = 0.01%:', iapars[1])

    ----------------------------------------------------------------------

    """
    if not isinstance(thresholds, list) and not isinstance(thresholds, float):
        raise TypeError(
            f"Expected thresholds of type List[float] or float but got: {type(thresholds)}"
        )

    morph = np.squeeze(np.array(morph))
    if isinstance(thresholds, float):
        return len(np.where(morph >= thresholds)[0]) / morph.shape[0]
    if isinstance(thresholds, list):
        resutls: List[float] = []
        for thres in thresholds:
            resutls.append(len(np.where(morph <= thres)[0]) / morph.shape[0])
        return resutls
