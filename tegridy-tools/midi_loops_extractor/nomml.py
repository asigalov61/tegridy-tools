
from __future__ import annotations

import numpy as np
from symusic import Score


def get_metric_depth(time, tpq, max_depth=6):
    for i in range(max_depth):
        period = tpq / int(2 ** i)
        if time % period == 0:
            return 2 * i
    for i in range(max_depth):
        period = tpq * 2 / (int(2 ** i) * 3)
        if time % period == 0:
            return 2 * i + 1
    return max_depth * 2


def get_median_metric_depth(path):
    mf = Score(path)
    median_metric_depths = []
    for track in mf.tracks:
        metric_depths = [get_metric_depth(event.time, mf.tpq) for event in track.notes]
        if len(metric_depths) > 0:
            median_metric_depths.append(int(np.median(metric_depths)))
    return path, median_metric_depths
