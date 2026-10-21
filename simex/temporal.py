"""Sampled temporal predicates over trusted object traces, never policy state."""
import math

MAX_SAMPLE_GAP = .30
EPSILON = 1e-6


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _height(obj):
    position = obj.get('position') if isinstance(obj, dict) else None
    if (not isinstance(position, (list, tuple)) or len(position) != 3
            or not all(_finite(value) for value in position)):
        return None
    return float(position[2])


def sustained_lift(name, obj, final_time, samples, minimum, seconds):
    """Require elevated measurements covering the entire final time window.

    A sample at/before the start and the actual final metric bracket the window.
    All intervening measurements must be elevated, with at least three points
    and no gap above 0.30 s. This measures sustained sampled elevation, not
    contact, grasp attachment, speed, or continuous-time behavior between samples.
    """
    initial = obj.get('initial_height') if isinstance(obj, dict) else None
    final_height = _height(obj)
    if (not all(_finite(value) for value in (initial, final_time, minimum, seconds))
            or seconds <= 0 or minimum <= 0 or final_height is None
            or not isinstance(samples, (list, tuple)) or len(samples) < 2):
        return False
    start = final_time - seconds
    points = []
    previous = None
    for sample in samples:
        if not isinstance(sample, dict) or not _finite(sample.get('sim_time')):
            return False
        time = sample['sim_time']
        if time > final_time + EPSILON or (previous is not None and time <= previous):
            return False
        previous = time
        objects = sample.get('objects', {})
        height = _height(objects.get(name)) if isinstance(objects, dict) else None
        points.append((time, height))
    preceding = [index for index, (time, _) in enumerate(points) if time <= start + EPSILON]
    if not preceding:
        return False
    window = points[preceding[-1]:]
    if final_time - window[-1][0] > EPSILON:
        window.append((final_time, final_height))
    if (len(window) < 3 or any(b[0] - a[0] > MAX_SAMPLE_GAP + EPSILON
                               for a, b in zip(window, window[1:]))):
        return False
    threshold = initial + minimum - EPSILON
    return final_height >= threshold and all(height is not None and height >= threshold
                                            for _, height in window)
