import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.ndimage
import tensorflow as tf


@dataclass
class DataConfig:
    manifest_path: str
    split_path: str
    image_size: int = 64
    planet_channels: int = 8
    hec_channels: int = 1
    batch_size: int = 16
    buffer_size: int = 1024
    seed: int = 42


def load_manifest_and_split(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(cfg.manifest_path)
    required = ["UUID", "date", "planet_path", "hecras_path", "qmask_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required manifest columns: {missing}")

    split = json.loads(Path(cfg.split_path).read_text())
    test_uuids = set(split["test_uuids"])
    dev_uuids = set(split["dev_uuids"])

    dev_df = df[df["UUID"].isin(dev_uuids)].copy()
    test_df = df[df["UUID"].isin(test_uuids)].copy()
    if len(test_df) == 0:
        raise ValueError("No held-out test rows matched split UUIDs.")
    return df, dev_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _minmax_norm(img):
    img = img.astype(np.float32)
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi) if ma > mi else np.zeros_like(img, dtype=np.float32)


def _fix_channels(arr, target_ch):
    ch = arr.shape[2]
    if ch < target_ch:
        pad = np.zeros((arr.shape[0], arr.shape[1], target_ch - ch), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=-1)
    elif ch > target_ch:
        arr = arr[..., :target_ch]
    return arr


def _pad_to_min_size(arr, size):
    h, w = arr.shape[:2]
    ph, pw = max(size - h, 0), max(size - w, 0)
    if ph == 0 and pw == 0:
        return arr
    top, bottom = ph // 2, ph - (ph // 2)
    left, right = pw // 2, pw - (pw // 2)
    return np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="constant")

def _center_crop(arr, size):
    arr = _pad_to_min_size(arr, size)
    h, w = arr.shape[:2]
    y0 = max((h - size) // 2, 0)
    x0 = max((w - size) // 2, 0)
    return arr[y0:y0 + size, x0:x0 + size]


def _random_crop(arr, size, rng):
    arr = _pad_to_min_size(arr, size)
    h, w = arr.shape[:2]
    if h <= size or w <= size:
        return _center_crop(arr, size)
    y0 = rng.integers(0, h - size + 1)
    x0 = rng.integers(0, w - size + 1)
    return arr[y0:y0 + size, x0:x0 + size]


def _load_tif(path):
    import tifffile
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def preprocess_sample(planet_path, hec_path, qmask_path, cfg: DataConfig, training=False):
    planet = _load_tif(planet_path.decode())
    hec = _load_tif(hec_path.decode())
    qmask = _load_tif(qmask_path.decode())

    planet = _fix_channels(planet, cfg.planet_channels)
    hec = _fix_channels(hec, cfg.hec_channels)
    if hec.shape[:2] != planet.shape[:2]:
        z = (planet.shape[0] / hec.shape[0], planet.shape[1] / hec.shape[1], 1)
        hec = scipy.ndimage.zoom(hec, z, order=1)
    if qmask.shape[:2] != planet.shape[:2]:
        z = (planet.shape[0] / qmask.shape[0], planet.shape[1] / qmask.shape[1], 1)
        qmask = scipy.ndimage.zoom(qmask, z, order=0)

    x = np.concatenate([_minmax_norm(planet), (hec > 0).astype(np.float32)], axis=-1)
    y = (qmask > 0).astype(np.float32)[..., :1]

    if training:
        rng = np.random.default_rng()
        xy = np.concatenate([x, y], axis=-1)
        xy = _random_crop(xy, cfg.image_size, rng)
        x, y = xy[..., :-1], xy[..., -1:]
    else:
        x = _center_crop(x, cfg.image_size)
        y = _center_crop(y, cfg.image_size)

    return x.astype(np.float32), y.astype(np.float32)


def _wrap_tf(planet, hec, qmask, cfg: DataConfig, training=False):
    x, y = tf.numpy_function(
        lambda a, b, c: preprocess_sample(a, b, c, cfg, training),
        [planet, hec, qmask], [tf.float32, tf.float32],
    )
    x.set_shape((cfg.image_size, cfg.image_size, cfg.planet_channels + cfg.hec_channels))
    y.set_shape((cfg.image_size, cfg.image_size, 1))
    return x, y


def make_dataset(df: pd.DataFrame, cfg: DataConfig, training=False):
    ds = tf.data.Dataset.from_tensor_slices((df["planet_path"].values, df["hecras_path"].values, df["qmask_path"].values))
    if training:
        ds = ds.shuffle(max(cfg.buffer_size, len(df)), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, h, q: _wrap_tf(p, h, q, cfg, training), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
