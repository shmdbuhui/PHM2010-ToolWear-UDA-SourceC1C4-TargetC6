"""Verify the two completed C1-to-C4 alignment ablation runs."""

import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]


def load_run(scale):
    tag = f"align_scale{scale}_seed42"
    ckpt_dir = ROOT / "artifacts" / "checkpoints" / f"DAREGRAM_bs128_ep50_lr1e-3_{tag}"
    log_path = next(ckpt_dir.glob("*.log"))
    weight_path = next(ckpt_dir.glob("*.pth"))
    log = log_path.read_text(encoding="utf-8")
    csv_path = ROOT / "visualization" / f"DAREGRAM_{tag}" / f"c1_tgt-c4_{tag}.csv"
    rows = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    with csv_path.open(encoding="utf-8") as stream:
        assert stream.readline().strip() == "sample_index,y_true_vb,y_pred_vb"
    assert rows.shape == (315, 3)
    assert np.array_equal(rows[:, 0], np.arange(1, 316))
    true, pred = rows[:, 1], rows[:, 2]
    matches = lambda pattern: re.findall(pattern, log)
    result = {
        "scale": scale,
        "tag": tag,
        "initial_sha256": matches(r"Initial model SHA256: ([0-9a-f]+)")[0],
        "batch_hashes": matches(r"Batch order SHA256: source=([0-9a-f]+) target=([0-9a-f]+)"),
        "bn_batches": [int(x) for x in matches(r"Backbone BN batches tracked: (\d+)")],
        "source_mse": [float(x) for x in matches(r"Train-Regressor\(MSE\): ([0-9.e+-]+)")],
        "total_loss": [float(x) for x in matches(r"Train-Total Loss: ([0-9.e+-]+)")],
        "gram_loss": [float(x) for x in matches(r"Train-DARE-GRAM Loss: ([0-9.e+-]+)")],
        "weighted_alignment": [float(x) for x in matches(r"Train-Weighted Alignment: ([0-9.e+-]+)")],
        "alignment_gradient_norms": [float(x) for x in matches(
            r"Alignment gradient norm, source=([0-9.e+-]+), target=[0-9.e+-]+"
        )],
        "target_alignment_gradient_norms": [float(x) for x in matches(
            r"Alignment gradient norm, source=[0-9.e+-]+, target=([0-9.e+-]+)"
        )],
        "metrics": {
            "mae": mean_absolute_error(true, pred),
            "rmse": math.sqrt(mean_squared_error(true, pred)),
            "r2": r2_score(true, pred),
            "mape_percent": float(np.mean(np.abs((true - pred) / np.maximum(np.abs(true), 1e-8))) * 100),
        },
        "log": str(log_path),
        "weight": str(weight_path),
        "predictions": str(csv_path),
    }
    for key in ("batch_hashes", "bn_batches", "source_mse", "total_loss", "gram_loss", "weighted_alignment"):
        assert len(result[key]) == 50, (tag, key, len(result[key]))
    for epoch, (mse, total, gram, weighted) in enumerate(zip(
        result["source_mse"], result["total_loss"], result["gram_loss"], result["weighted_alignment"]
    ), start=1):
        tradeoff = 2 / (1 + math.exp(-10 * (epoch - 1) / 49)) - 1
        assert abs(weighted - scale * tradeoff * gram) < 0.001
        assert abs(total - mse - weighted) < 0.002
    assert result["bn_batches"] == list(range(4, 201, 4))
    assert len(result["alignment_gradient_norms"]) == 1
    assert len(result["target_alignment_gradient_norms"]) == 1
    assert weight_path.stat().st_size > 1_000_000
    return result, true


def main():
    a, true_a = load_run(0)
    b, true_b = load_run(1)
    assert np.array_equal(true_a, true_b)
    assert a["initial_sha256"] == b["initial_sha256"]
    assert a["batch_hashes"] == b["batch_hashes"]
    assert a["source_mse"][0] == b["source_mse"][0]
    assert a["gram_loss"][0] == b["gram_loss"][0]
    assert all(x == 0 for x in a["weighted_alignment"])
    assert any(x > 0 for x in b["weighted_alignment"][1:])
    assert a["alignment_gradient_norms"] == [0]
    assert a["target_alignment_gradient_norms"] == [0]
    assert b["alignment_gradient_norms"][0] > 0
    assert b["target_alignment_gradient_norms"][0] > 0
    result = {
        "matching_initial_weights": True,
        "matching_source_and_target_batch_orders_all_50_epochs": True,
        "same_c4_evaluation_labels": True,
        "a": a,
        "b": b,
        "b_minus_a": {key: b["metrics"][key] - a["metrics"][key] for key in a["metrics"]},
    }
    out = ROOT / "artifacts" / "ablation_audit_seed42.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"a": a["metrics"], "b": b["metrics"], "b_minus_a": result["b_minus_a"]}, indent=2))
    print(f"Verified: {out}")


if __name__ == "__main__":
    main()
