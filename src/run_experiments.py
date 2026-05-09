import argparse
import itertools
import time
from datetime import datetime
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
    return {"bce_dice": bce_dice_loss, "focal_dice": focal_dice_loss, "tversky": tversky_loss}[name]


def eval_threshold_metrics(model, val_df, dcfg, thresholds):
    ds = make_dataset(val_df, dcfg, training=False)
    y_true, y_prob = [], []
    for xb, yb in ds:
        y_true.append(yb.numpy())
        y_prob.append(model.predict(xb, verbose=0))
    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)

    best_t, best_d, best_i = 0.5, -1.0, -1.0
    for t in thresholds:
        yp = (y_prob > t).astype(np.float32)
        inter = np.sum(y_true * yp)
        union = np.sum(y_true) + np.sum(yp) - inter
        dice = (2.0 * inter + 1e-7) / (np.sum(y_true) + np.sum(yp) + 1e-7)
        iou = (inter + 1e-7) / (union + 1e-7)
        if dice > best_d:
            best_t, best_d, best_i = float(t), float(dice), float(iou)
    return best_t, best_d, best_i


def read_csv_if_exists(path, cols):
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)


def save_fold_result(path, row):
    cols = ["exp_id", "fold", "epochs", "loss", "crop_size", "model_variant", "val_dice", "val_iou", "best_threshold", "elapsed_sec"]
    df = read_csv_if_exists(path, cols)
    m = (df.get("exp_id", pd.Series(dtype=int)) == row["exp_id"]) & (df.get("fold", pd.Series(dtype=int)) == row["fold"])
    if len(df):
        df = df.loc[~m].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.sort_values(["exp_id", "fold"], inplace=True)
    df.to_csv(path, index=False)


def save_experiment_results(results_dir, row):
    cols = ["exp_id", "epochs", "loss", "crop_size", "model_variant", "mean_val_dice", "std_val_dice", "mean_threshold", "completed_folds"]
    log_path = results_dir / "experiment_log.csv"
    df = read_csv_if_exists(log_path, cols)
    m = (df.get("exp_id", pd.Series(dtype=int)) == row["exp_id"])
    if len(df):
        df = df.loc[~m].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.sort_values(["exp_id"], inplace=True)
    df.to_csv(log_path, index=False)
    df.sort_values(["mean_val_dice"], ascending=False).reset_index(drop=True).to_csv(results_dir / "leaderboard_validation.csv", index=False)


def save_failed_run(path, row):
    cols = ["exp_id", "fold", "epochs", "loss", "crop_size", "model_variant", "stage", "error"]
    df = read_csv_if_exists(path, cols)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def main(rerun=False):
    t0 = time.time()
    supported_crop_sizes = {64}
    search_cfg = yaml.safe_load(Path("configs/experiments/search_space.yaml").read_text())
    exp_cfg = yaml.safe_load(Path(search_cfg["base_experiment_config"]).read_text())
    paths = yaml.safe_load(Path(exp_cfg["paths_config"]).read_text())

    results_dir = Path(search_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_path = results_dir / "fold_results.csv"
    failed_path = results_dir / "failed_runs.csv"
    log_path = results_dir / "experiment_log.csv"
    leaderboard_path = results_dir / "leaderboard_validation.csv"
    final_path = results_dir / "final_test_results.csv"

    dcfg = DataConfig(manifest_path=paths["clean_manifest"], split_path=paths["split_file"], **exp_cfg["data"])
    _, dev_df, test_df = load_manifest_and_split(dcfg)

    grid = list(itertools.product(search_cfg["search_space"]["epochs"], search_cfg["search_space"]["losses"], search_cfg["search_space"]["crop_sizes"], search_cfg["search_space"]["model_variants"]))[: search_cfg.get("max_configs", 12)]
    total_cfg = len(grid)
    thresholds = search_cfg["search_space"]["thresholds"]
    n_splits = exp_cfg["training"]["n_splits"]
    gkf = GroupKFold(n_splits=n_splits)
    existing_fold_df = read_csv_if_exists(fold_path, ["exp_id", "fold"])

    for exp_id, (epochs, loss_name, crop, variant) in enumerate(grid, start=1):
        print(f"\n[EXP-START] {exp_id}/{total_cfg} | exp_id={exp_id} | variant={variant} | loss={loss_name} | epochs={epochs} | crop={crop} | thresholds={thresholds} | batch_size={dcfg.batch_size} | image_size={int(crop)} | start={datetime.utcnow().isoformat()}Z")
        if int(crop) not in supported_crop_sizes:
            print(f"[SKIP] exp_id={exp_id}/{total_cfg} crop={crop} unsupported")
            save_failed_run(failed_path, {"exp_id": exp_id, "fold": "", "epochs": epochs, "loss": loss_name, "crop_size": crop, "model_variant": variant, "stage": "config", "error": "unsupported crop size"})
            continue

        dcfg_exp = DataConfig(**{**dcfg.__dict__, "image_size": int(crop)})
        for fold, (tr, va) in enumerate(gkf.split(dev_df, groups=dev_df["UUID"]), start=1):
            tr_df, va_df = dev_df.iloc[tr], dev_df.iloc[va]
            print(f"[FOLD-START] exp_id={exp_id} fold={fold}/{n_splits} train_samples={len(tr_df)} val_samples={len(va_df)} train_uuids={tr_df['UUID'].nunique()} val_uuids={va_df['UUID'].nunique()}")
            done = len(existing_fold_df) and ((existing_fold_df["exp_id"] == exp_id) & (existing_fold_df["fold"] == fold)).any()
            if done and not rerun:
                print(f"[RESUME-SKIP] exp_id={exp_id} fold={fold}/{n_splits}")
                continue

            f0 = time.time()
            try:
                model = build_unet_binary(input_shape=(dcfg_exp.image_size, dcfg_exp.image_size, 9), **exp_cfg["model"])
                model.compile(optimizer=tf.keras.optimizers.Adam(exp_cfg["training"]["lr"]), loss=get_loss(loss_name), metrics=[dice_coef, iou_metric])
                model.fit(make_dataset(tr_df, dcfg_exp, True), validation_data=make_dataset(va_df, dcfg_exp, False), epochs=int(epochs), verbose=0)
                best_t, best_d, best_i = eval_threshold_metrics(model, va_df, dcfg_exp, thresholds)
                elapsed = time.time() - f0
                save_fold_result(fold_path, {"exp_id": exp_id, "fold": fold, "epochs": epochs, "loss": loss_name, "crop_size": crop, "model_variant": variant, "val_dice": best_d, "val_iou": best_i, "best_threshold": best_t, "elapsed_sec": round(elapsed, 2)})
                existing_fold_df = read_csv_if_exists(fold_path, ["exp_id", "fold"])
                print(f"[FOLD-DONE] val_dice={best_d:.6f} val_iou={best_i:.6f} threshold={best_t:.2f} elapsed={elapsed:.1f}s saved={fold_path}")
            except Exception as e:
                save_failed_run(failed_path, {"exp_id": exp_id, "fold": fold, "epochs": epochs, "loss": loss_name, "crop_size": crop, "model_variant": variant, "stage": "fold", "error": str(e)})
                print(f"[FOLD-FAIL] exp_id={exp_id} fold={fold}/{n_splits} err={e}")

        exp_folds = read_csv_if_exists(fold_path, ["exp_id", "val_dice", "best_threshold"])
        exp_folds = exp_folds[exp_folds["exp_id"] == exp_id]
        if len(exp_folds) < n_splits:
            continue

        exp_row = {"exp_id": exp_id, "epochs": epochs, "loss": loss_name, "crop_size": crop, "model_variant": variant, "mean_val_dice": float(exp_folds["val_dice"].mean()), "std_val_dice": float(exp_folds["val_dice"].std(ddof=0)), "mean_threshold": float(exp_folds["best_threshold"].mean()), "completed_folds": int(len(exp_folds))}
        save_experiment_results(results_dir, exp_row)
        current_best = read_csv_if_exists(leaderboard_path, ["exp_id"]).iloc[0]["exp_id"] == exp_id
        print(f"[EXP-DONE] mean_val_dice={exp_row['mean_val_dice']:.6f} std_val_dice={exp_row['std_val_dice']:.6f} mean_threshold={exp_row['mean_threshold']:.4f} current_best={bool(current_best)}")

    exp_df = read_csv_if_exists(log_path, ["exp_id"])
    if len(exp_df) == 0:
        print("No completed experiments yet. Resume with: python src/run_experiments.py")
        return

    best = exp_df.sort_values(["mean_val_dice"], ascending=False).iloc[0].to_dict()
    best_cfg = {"epochs": int(best["epochs"]), "loss": best["loss"], "crop_size": int(best["crop_size"]), "model_variant": best["model_variant"], "threshold": float(best["mean_threshold"]), "selection_metric": "mean GroupKFold validation Dice"}
    (results_dir / "best_config.yaml").write_text(yaml.safe_dump(best_cfg, sort_keys=False))

    dcfg_best = DataConfig(**{**dcfg.__dict__, "image_size": int(best_cfg["crop_size"])})
    model = build_unet_binary(input_shape=(dcfg_best.image_size, dcfg_best.image_size, 9), **exp_cfg["model"])
    model.compile(optimizer=tf.keras.optimizers.Adam(exp_cfg["training"]["lr"]), loss=get_loss(best_cfg["loss"]), metrics=[dice_coef, iou_metric])
    model.fit(make_dataset(dev_df, dcfg_best, True), epochs=best_cfg["epochs"], verbose=0)

    y_true, y_prob = [], []
    for xb, yb in make_dataset(test_df, dcfg_best, False):
        y_true.append(yb.numpy())
        y_prob.append(model.predict(xb, verbose=0))
    y_true, y_prob = np.concatenate(y_true, 0), np.concatenate(y_prob, 0)
    yp = (y_prob > best_cfg["threshold"]).astype(np.float32)
    inter = np.sum(y_true * yp)
    union = np.sum(y_true) + np.sum(yp) - inter
    dice = (2.0 * inter + 1e-7) / (np.sum(y_true) + np.sum(yp) + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)
    pd.DataFrame([{"heldout_samples": int(len(test_df)), "threshold": best_cfg["threshold"], "test_iou": float(iou), "test_dice": float(dice)}]).to_csv(final_path, index=False)

    print("\n[FINAL]")
    print(f"best_experiment={best_cfg}")
    print(f"best_mean_val_dice={best['mean_val_dice']:.6f}")
    print(f"final_test_iou={iou:.6f} final_test_dice={dice:.6f}")
    print(f"outputs: fold={fold_path} log={log_path} leaderboard={leaderboard_path} failed={failed_path} final_test={final_path}")
    print(f"elapsed_total={time.time()-t0:.1f}s")
    print("Resume after disconnect: python src/run_experiments.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    main(rerun=args.rerun)
