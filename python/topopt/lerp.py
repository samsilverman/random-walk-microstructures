from __future__ import annotations


def lerp(l0: float, l1: float, step: int, num_steps: int) -> float:
    """Linearly interpolate between two values over a fixed number of steps.

    Parameters
    ----------
    l0 : float
        Start value.
    l1 : float
        End value.
    step : int
        Current step.
    num_steps : int
        Total number of steps in the schedule.

    Returns
    -------
    value : float
        Interpolated value.

    """
    if num_steps <= 1:
        return float(l1)

    t = step / (num_steps - 1)

    # Clamp t to [0,1] so we don't extrapolate outside [l0,l1].
    t = max(0.0, min(1.0, float(t)))

    return l0 + (l1 - l0) * t
