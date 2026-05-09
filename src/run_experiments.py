import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import GroupKFold

from data import DataConfig, load_manifest_and_split, make_dataset
from losses import bce_dice_loss, dice_coef, iou_metric
from models import build_unet_binary


def focal_dice_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
    focal = -alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
    return tf.reduce_mean(focal) + (1.0 - dice_coef(y_true, y_pred))


def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    t = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - t


def get_loss(name):
    if name == "bce_dice":
        return bce_dice_loss
    if name == "focal_dice":
        return focal_dice_loss
    if name == "tversky":
        return tversky_loss
    raise ValueError(name)


def evaluate_threshold(model, val_df, dcfg, thresholds):
    ds = make_dataset(val_df, dcfg, training=False)
    y_true_all, y_prob_all = [], []
    for xb, yb in ds:
        yp = model.predict(xb, verbose=0)
        y_true_all.append(yb.numpy())
        y_prob_all.append(yp)
    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)

    best_t, best_d = 0.5, -1.0
    for t in thresholds:
        yp = (y_prob > t).astype(np.float32)
        inter = np.sum(y_true * yp)
        denom = np.sum(y_true) + np.sum(yp)
        d = (2.0 * inter + 1e-7) / (denom + 1e-7)
        if d > best_d:
            best_t, best_d = float(t), float(d)
    return best_t, best_d


def main():
    search_cfg = yaml.safe_load(Path("configs/experiments/search_space.yaml").read_text())
    exp_cfg = yaml.safe_load(Path(search_cfg["base_experiment_config"]).read_text())
    paths = yaml.safe_load(Path(exp_cfg["paths_config"]).read_text())

    results_dir = Path(search_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    dcfg = DataConfig(manifest_path=paths["clean_manifest"], split_path=paths["split_file"], **exp_cfg["data"])
    _, dev_df, test_df = load_manifest_and_split(dcfg)

    grid = list(itertools.product(
        search_cfg["search_space"]["epochs"],
        search_cfg["search_space"]["losses"],
        search_cfg["search_space"]["crop_sizes"],
        search_cfg["search_space"]["model_variants"],
    ))[: search_cfg.get("max_configs", 12)]

    fold_rows, exp_rows = [], []
    thresholds = search_cfg["search_space"]["thresholds"]
    gkf = GroupKFold(n_splits=exp_cfg["training"]["n_splits"])

    for exp_id, (epochs, loss_name, crop, variant) in enumerate(grid, start=1):
        dcfg_exp = DataConfig(**{**dcfg.__dict__, "image_size": int(crop)})
        fold_dice, fold_thr = [], []

        for fold, (tr, va) in enumerate(gkf.split(dev_df, groups=dev_df["UUID"]), start=1):
            tr_df, va_df = dev_df.iloc[tr], dev_df.iloc[va]
            model = build_unet_binary(
                input_shape=(dcfg_exp.image_size, dcfg_exp.image_size, 9),
                base_filters=exp_cfg["model"]["base_filters"],
                use_bn=exp_cfg["model"]["use_bn"],
                dropout=exp_cfg["model"]["dropout"],
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(exp_cfg["training"]["lr"]),
                loss=get_loss(loss_name),
                metrics=[dice_coef, iou_metric],
            )
            model.fit(make_dataset(tr_df, dcfg_exp, True), validation_data=make_dataset(va_df, dcfg_exp, False), epochs=int(epochs), verbose=0)
            best_t, best_d = evaluate_threshold(model, va_df, dcfg_exp, thresholds)
            fold_dice.append(best_d)
            fold_thr.append(best_t)
            fold_rows.append({"exp_id": exp_id, "fold": fold, "epochs": epochs, "loss": loss_name, "crop_size": crop, "model_variant": variant, "val_dice": best_d, "best_threshold": best_t})

        exp_rows.append({"exp_id": exp_id, "epochs": epochs, "loss": loss_name, "crop_size": crop, "model_variant": variant, "mean_val_dice": float(np.mean(fold_dice)), "std_val_dice": float(np.std(fold_dice)), "mean_threshold": float(np.mean(fold_thr))})

    fold_df = pd.DataFrame(fold_rows)
    exp_df = pd.DataFrame(exp_rows).sort_values(["mean_val_dice"], ascending=False).reset_index(drop=True)

    fold_df.to_csv(results_dir / "fold_results.csv", index=False)
    exp_df.to_csv(results_dir / "leaderboard_validation.csv", index=False)
    exp_df.to_csv(results_dir / "experiment_log.csv", index=False)

    best = exp_df.iloc[0].to_dict()
    best_cfg = {
        "epochs": int(best["epochs"]),
        "loss": best["loss"],
        "crop_size": int(best["crop_size"]),
        "model_variant": best["model_variant"],
        "threshold": float(best["mean_threshold"]),
        "selection_metric": "mean GroupKFold validation Dice",
    }
    (results_dir / "best_config.yaml").write_text(yaml.safe_dump(best_cfg, sort_keys=False))

    # final train on full dev, held-out test once
    dcfg_best = DataConfig(**{**dcfg.__dict__, "image_size": int(best_cfg["crop_size"])})
    model = build_unet_binary(input_shape=(dcfg_best.image_size, dcfg_best.image_size, 9), **exp_cfg["model"])
    model.compile(optimizer=tf.keras.optimizers.Adam(exp_cfg["training"]["lr"]), loss=get_loss(best_cfg["loss"]), metrics=[dice_coef, iou_metric])
    model.fit(make_dataset(dev_df, dcfg_best, True), epochs=best_cfg["epochs"], verbose=0)

    # compute held-out metrics with selected threshold
    y_true_all, y_prob_all = [], []
    for xb, yb in make_dataset(test_df, dcfg_best, False):
        y_true_all.append(yb.numpy())
        y_prob_all.append(model.predict(xb, verbose=0))
    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    yp = (y_prob > best_cfg["threshold"]).astype(np.float32)
    inter = np.sum(y_true * yp)
    union = np.sum(y_true) + np.sum(yp) - inter
    dice = (2.0 * inter + 1e-7) / (np.sum(y_true) + np.sum(yp) + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)

    final_df = pd.DataFrame([{
        "heldout_samples": int(len(test_df)), "threshold": best_cfg["threshold"], "test_iou": float(iou), "test_dice": float(dice)
    }])
    final_df.to_csv(results_dir / "final_test_results.csv", index=False)


if __name__ == "__main__":
    main()
