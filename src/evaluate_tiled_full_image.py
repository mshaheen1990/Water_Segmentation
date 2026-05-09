import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from data import DataConfig, load_manifest_and_split, _fix_channels, _minmax_norm, _load_tif
from losses import bce_dice_loss, dice_coef, iou_metric


def build_full_input(planet_path, hec_path, qmask_path, cfg):
    import scipy.ndimage

    planet = _load_tif(planet_path)
    hec = _load_tif(hec_path)
    qmask = _load_tif(qmask_path)
    planet = _fix_channels(planet, cfg.planet_channels)
    hec = _fix_channels(hec, cfg.hec_channels)
    if hec.shape[:2] != planet.shape[:2]:
        hec = scipy.ndimage.zoom(hec, (planet.shape[0]/hec.shape[0], planet.shape[1]/hec.shape[1], 1), order=1)
    if qmask.shape[:2] != planet.shape[:2]:
        qmask = scipy.ndimage.zoom(qmask, (planet.shape[0]/qmask.shape[0], planet.shape[1]/qmask.shape[1], 1), order=0)
    x = np.concatenate([_minmax_norm(planet), (hec > 0).astype(np.float32)], axis=-1)
    y = (qmask > 0).astype(np.float32)[..., :1]
    return x, y


def tiled_predict(model, x, tile_size=64, stride=64):
    h, w, c = x.shape
    ph = int(np.ceil(h / stride) * stride)
    pw = int(np.ceil(w / stride) * stride)
    pad = np.zeros((ph, pw, c), dtype=np.float32)
    pad[:h, :w] = x

    prob = np.zeros((ph, pw, 1), dtype=np.float32)
    cnt = np.zeros((ph, pw, 1), dtype=np.float32)
    for y0 in range(0, ph - tile_size + 1, stride):
        for x0 in range(0, pw - tile_size + 1, stride):
            tile = pad[y0:y0+tile_size, x0:x0+tile_size]
            pred = model.predict(tile[None, ...], verbose=0)[0]
            prob[y0:y0+tile_size, x0:x0+tile_size] += pred
            cnt[y0:y0+tile_size, x0:x0+tile_size] += 1.0
    prob = prob / np.maximum(cnt, 1.0)
    return prob[:h, :w]


def main(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    dcfg = DataConfig(manifest_path=paths["clean_manifest"], split_path=paths["split_file"], **cfg["data"])
    _, _, test_df = load_manifest_and_split(dcfg)

    model_path = cfg.get("eval", {}).get("model_path", str(Path(cfg["output_dir"]) / "unet_earlyfusion_final.keras"))
    model = tf.keras.models.load_model(model_path, custom_objects={"bce_dice_loss": bce_dice_loss, "iou_metric": iou_metric, "dice_coef": dice_coef})

    rows = []
    threshold = 0.5
    for _, r in test_df.iterrows():
        x, y = build_full_input(r["planet_path"], r["hecras_path"], r["qmask_path"], dcfg)
        p = tiled_predict(model, x, tile_size=dcfg.image_size, stride=dcfg.image_size)
        pb = (p > threshold).astype(np.float32)
        inter = np.sum(y * pb)
        union = np.sum(y) + np.sum(pb) - inter
        dice = (2 * inter + 1e-7) / (np.sum(y) + np.sum(pb) + 1e-7)
        iou = (inter + 1e-7) / (union + 1e-7)
        rows.append({"UUID": r["UUID"], "date": r["date"], "iou": float(iou), "dice": float(dice)})

    df = pd.DataFrame(rows)
    out = Path("results/tiled_full_image_test_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{"heldout_samples": len(df), "mean_iou": float(df['iou'].mean()), "mean_dice": float(df['dice'].mean())}])
    pd.concat([summary, pd.DataFrame([{}]), df], ignore_index=True).to_csv(out, index=False)
    print(f"Saved tiled full-image results to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/unet_earlyfusion.yaml")
    args = ap.parse_args()
    main(args.config)
