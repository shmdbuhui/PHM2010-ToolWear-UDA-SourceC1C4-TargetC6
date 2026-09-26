"""Cut-aware soft trend loss and read-only physical audit in original VB units."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def adjacent_positions(tools, cuts):
    """Return unique, real cut i -> i+1 pairs within each tool in cut order."""
    if len(tools) != len(cuts):
        raise ValueError("Tool/cut lengths differ")
    buckets = {}
    for pos, (tool, cut) in enumerate(zip(tools, cuts)):
        key = (str(tool), int(cut))
        buckets.setdefault(key, []).append(pos)
    left, right = [], []
    for (tool, cut), positions in sorted(buckets.items()):
        if len(positions) == 1 and len(buckets.get((tool, cut + 1), [])) == 1:
            left.append(positions[0])
            right.append(buckets[(tool, cut + 1)][0])
    return left, right


def source_delta(labels, cuts):
    """Fixed heuristic: 25% of median absolute adjacent source-label change."""
    labels = np.asarray(labels, dtype=np.float64)
    left, right = adjacent_positions(["source"] * len(cuts), cuts)
    if not left or not np.isfinite(labels).all():
        raise ValueError("Source labels need finite, adjacent cuts")
    changes = np.abs(labels[left] - labels[right])
    return float(0.25 * np.median(changes)), len(left)


def trend_loss(predictions, tools, cuts, delta):
    left, right = adjacent_positions(tools, cuts)
    if not left:
        return predictions.sum() * 0.0, 0
    decline = predictions.reshape(-1)[left] - predictions.reshape(-1)[right]
    return F.relu(decline - delta).square().mean(), len(left)


def pava_nondecreasing(values):
    """Equal-weight L2 projection onto nondecreasing sequences (PAVA)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("PAVA requires a nonempty finite one-dimensional prediction")
    levels, weights, starts, ends = [], [], [], []
    for index, value in enumerate(values):
        levels.append(float(value))
        weights.append(1)
        starts.append(index)
        ends.append(index + 1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            total = weights[-2] + weights[-1]
            merged = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total
            levels[-2] = merged
            weights[-2] = total
            ends[-2] = ends[-1]
            levels.pop()
            weights.pop()
            starts.pop()
            ends.pop()
    result = np.empty_like(values)
    for level, start, end in zip(levels, starts, ends):
        result[start:end] = level
    if np.any(np.diff(result) < -1e-8):
        raise AssertionError("PAVA projection is not nondecreasing")
    return result


def physical_audit(cuts, predictions, delta, source_max, expected=range(1, 316)):
    """Never transform predictions; adjacent statistics use only unique finite neighbors."""
    cuts = np.asarray(cuts)
    pred = np.asarray(predictions, dtype=np.float64)
    if len(cuts) != len(pred) or not np.isfinite(delta) or delta < 0:
        raise ValueError("Invalid physical audit input")
    if not all(float(c).is_integer() for c in cuts):
        raise ValueError("Cut IDs must be integers")
    cuts = cuts.astype(int)
    count = Counter(cuts.tolist())
    missing = sorted(set(expected) - set(cuts.tolist()))
    duplicate = sorted(c for c, n in count.items() if n > 1)
    order = np.argsort(cuts, kind="stable")
    sorted_cuts, sorted_pred = cuts[order], pred[order]
    lookup = {int(c): float(p) for c, p in zip(sorted_cuts, sorted_pred) if count[int(c)] == 1}
    rows = []
    declines, rises = [], []
    for cut, value in zip(sorted_cuts, sorted_pred):
        next_value = lookup.get(int(cut) + 1) if count[int(cut)] == 1 else None
        valid = next_value is not None and np.isfinite(value) and np.isfinite(next_value)
        step = float(next_value - value) if valid else np.nan
        if valid:
            declines.append(max(-step, 0.0))
            rises.append(max(step, 0.0))
        rows.append({"cut_index": int(cut), "pred_vb_raw": float(value),
                     "next_cut_index": int(cut + 1) if valid else np.nan,
                     "next_minus_current_vb": step,
                     "decline_vb": max(-step, 0.0) if valid else np.nan,
                     "decline_exceeds_delta": bool(valid and -step > delta),
                     "negative_prediction": bool(np.isfinite(value) and value < 0),
                     "above_source_label_max_oor": bool(np.isfinite(value) and value > source_max)})
    finite = pred[np.isfinite(pred)]
    result = {"observed_count": int(len(pred)), "unique_cut_count": int(len(count)),
              "missing_cuts": missing, "duplicate_cuts": duplicate,
              "missing_cut_count": len(missing), "duplicate_cut_count": len(duplicate),
              "nonfinite_count": int((~np.isfinite(pred)).sum()),
              "negative_count": int((finite < 0).sum()),
              "adjacent_pair_count": len(declines),
              "adjacent_decline_count": int(np.count_nonzero(np.asarray(declines) > 0)),
              "max_decline_vb": float(max(declines, default=0)),
              "decline_over_delta_count": int(np.count_nonzero(np.asarray(declines) > delta)),
              "max_upward_jump_vb": float(max(rises, default=0)),
              "prediction_min_vb": float(finite.min()) if len(finite) else None,
              "prediction_max_vb": float(finite.max()) if len(finite) else None,
              "prediction_range_vb": float(np.ptp(finite)) if len(finite) else None,
              "prediction_variance_vb2": float(np.var(finite)) if len(finite) else None,
              "source_label_max_vb": float(source_max),
              "above_source_max_oor_count": int((finite > source_max).sum()),
              "delta_vb": float(delta)}
    return pd.DataFrame(rows), result
