# preprocess.py
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import zoom
from typing import Optional, List

EXPECTED_COLS_ALL = ['Fx','Fy','Fz','Vx','Vy','Vz','aerms']
USE_AE_RMS = False
EXPECTED_INPUT_COLS = EXPECTED_COLS_ALL if USE_AE_RMS else ['Fx','Fy','Fz','Vx','Vy','Vz']

TARGET_SIZE = 128  # resize mat


def _extract_pass_idx(fname: str) -> Optional[int]:
    m = re.search(r'_(\d+)\.csv$', os.path.basename(fname))
    return int(m.group(1)) if m else None


def _load_wear_mean(wear_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(wear_csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    use = df[[cols['cut'], cols['flute_1'], cols['flute_2'], cols['flute_3']]].copy()
    use.columns = ['pass','flute_1','flute_2','flute_3']
    use['pass'] = pd.to_numeric(use['pass'], errors='coerce').round().astype('Int64')
    for f in ['flute_1','flute_2','flute_3']:
        use[f] = pd.to_numeric(use[f], errors='coerce')

    use['VB'] = use[['flute_1','flute_2','flute_3']].mean(axis=1)
    return use[['pass','VB']].dropna().astype({'pass': int}).sort_values('pass')


def _read_pass_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] != len(EXPECTED_COLS_ALL):
        raise ValueError(f"{os.path.basename(csv_path)}: {df.shape[1]} cols")
    df.columns = EXPECTED_COLS_ALL
    return df


def _crop_fixed_center(df: pd.DataFrame, input_cols: List[str], window_len: int) -> Optional[np.ndarray]:
    T = len(df)
    if T < window_len:
        return None
    mid = T // 2
    half = window_len // 2
    s = mid - half
    e = s + window_len
    return df[input_cols].values.T[:, s:e]  # (C, L)


def _stft_crop_and_resize(
    arr: np.ndarray,
    nperseg: int = 256,
    noverlap: int = 224,      # hop=32 => W~121 for L=4096
    fs: float = 50_000.0,
    fmax: float = 10_000.0,   # <-- freq crop (Hz)
    target: int = 128
) -> np.ndarray:
    """
    (C,L) -> STFT -> log -> freq crop(<=fmax) -> resize -> (C,128,128)
    """
    C, _ = arr.shape
    out = []

    for c in range(C):
        f, t, Zxx = signal.stft(
            arr[c],
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            boundary=None,
            padded=False
        )

        mag = np.log1p(np.abs(Zxx))  # (H, W)

        # # freq crop
        # if fmax is not None and fmax > 0:
        #     idx = np.where(f <= fmax)[0]
        #     if len(idx) < 2:
        #         raise ValueError(f"fmax too small or fs/nperseg mismatch: fmax={fmax}, df={fs/nperseg}")
        #     mag = mag[idx, :]  # (Hc, W)

        H, W = mag.shape
        zh = target / H
        zw = target / W
        mag_rs = zoom(mag, (zh, zw), order=1).astype(np.float32)
        out.append(mag_rs)

    return np.stack(out, axis=0)  # (C, 128, 128)


def _collect_pass_list(root: str, condition: str, wear_map: dict):
    cid = condition[1:]
    pass_dir = os.path.join(root, condition, condition)
    if not os.path.isdir(pass_dir):
        pass_dir = os.path.join(root, condition)
    files = sorted(glob.glob(os.path.join(pass_dir, f"c_{cid}_*.csv")))

    items = []
    for fp in files:
        pidx = _extract_pass_idx(fp)
        if pidx in wear_map:
            items.append((fp, pidx, wear_map[pidx]))
    return items


def build_and_save(
    root: str,
    conditions: List[str],
    out_dir: str,
    window_len: int,
    nperseg: int = 256,
    noverlap: int = 224,
    fs: float = 50_000.0,
    fmax: float = 10_000.0,
    target: int = 128,
):
    os.makedirs(out_dir, exist_ok=True)

    hop = nperseg - noverlap
    approx_W = 1 + max(0, (window_len - nperseg) // max(1, hop))
    print(f"[CFG] window_len={window_len}, fs={fs}, nperseg={nperseg}, noverlap={noverlap}, hop={hop}, approx_W~{approx_W}, fmax={fmax}, target={target}")

    for cond in conditions:
        print(f"\n===== CONDITION {cond} =====")

        wear_path = os.path.join(root, cond, f"{cond}_wear.csv")
        if not os.path.exists(wear_path):
            wear_path = os.path.join(root, f"{cond}_wear.csv")
        wear_df = _load_wear_mean(wear_path)
        wear_map = dict(zip(wear_df['pass'], wear_df['VB']))
        print(f"[INFO] wear entries: {len(wear_map)}")

        items = _collect_pass_list(root, cond, wear_map)
        print(f"[INFO] pass files matched: {len(items)}")

        X, y = [], []

        for fp, pidx, vb in items:
            print(f"  -> pass {pidx}: ", end="")

            try:
                df = _read_pass_df(fp)
            except Exception as e:
                print(f"READ FAIL ({e})")
                continue

            arr = _crop_fixed_center(df, EXPECTED_INPUT_COLS, window_len)
            if arr is None:
                print("CROP FAIL")
                continue

            img = _stft_crop_and_resize(
                arr,
                nperseg=nperseg,
                noverlap=noverlap,
                fs=fs,
                fmax=fmax,
                target=target
            )
            print(f"OK {img.shape}")

            X.append(img)
            y.append(vb)

        if not X:
            print("[WARN] no samples")
            continue

        X = np.stack(X, axis=0).astype(np.float32)
        y = np.array(y, dtype=np.float32)

        for split in ["train", "target", "unlabeled"]:
            out_path = os.path.join(out_dir, f"{split}_{cond}.npz")
            np.savez_compressed(out_path, samples=X, labels=y)
            print(f"[SAVE] {out_path}")

        print(f"       X: {X.shape} dtype={X.dtype}")
        print(f"       y: {y.shape} dtype={y.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="dataset")
    parser.add_argument("--conds", type=str, nargs="+", default=["c1","c4","c6"])
    parser.add_argument("--out", type=str, default="dataset")

    parser.add_argument("--window_len", type=int, default=4096)
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--noverlap", type=int, default=224)      
    parser.add_argument("--fs", type=float, default=50000.0)
    parser.add_argument("--fmax", type=float, default=20000.0)    
    parser.add_argument("--target", type=int, default=128)

    args = parser.parse_args()

    build_and_save(
        root=args.root,
        conditions=args.conds,
        out_dir=args.out,
        window_len=args.window_len,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        fs=args.fs,
        fmax=args.fmax,
        target=args.target,
    )
