import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import tensorflow as tf
import yaml
from sklearn.model_selection import GroupKFold

from data import DataConfig, _fix_channels, _load_tif, _minmax_norm, load_manifest_and_split
from losses import bce_dice_loss, dice_coef, iou_metric


def now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


def write_status(path, stage, config_id="", fold="", processed=0, elapsed=0.0):
    path.write_text(
        f"last_update={now()}\ncurrent_stage={stage}\ncurrent_config={config_id}\ncurrent_fold={fold}\n"
        f"processed_samples={processed}\nelapsed_sec={elapsed:.1f}\n"
    )


def blend_window(size, blending):
    if blending == "hann":
        h = np.hanning(size)
        w = np.outer(h, h)
        return (w / (w.max() + 1e-7))[..., None].astype(np.float32)
    if blending == "gaussian":
        x = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, x)
        w = np.exp(-(xx**2 + yy**2) / (2 * 0.3**2))
        return (w / (w.max() + 1e-7))[..., None].astype(np.float32)
    return np.ones((size, size, 1), dtype=np.float32)


def preprocess_full(row, dcfg):
    p = _fix_channels(_load_tif(row["planet_path"]), dcfg.planet_channels)
    h = _fix_channels(_load_tif(row["hecras_path"]), dcfg.hec_channels)
    q = _load_tif(row["qmask_path"])
    if h.shape[:2] != p.shape[:2]:
        h = scipy.ndimage.zoom(h, (p.shape[0] / h.shape[0], p.shape[1] / h.shape[1], 1), order=1)
    if q.shape[:2] != p.shape[:2]:
        q = scipy.ndimage.zoom(q, (p.shape[0] / q.shape[0], p.shape[1] / q.shape[1], 1), order=0)
    x = np.concatenate([_minmax_norm(p), (h > 0).astype(np.float32)], axis=-1)
    y = (q > 0).astype(np.uint8)[..., :1]
    return x, y, (h > 0).astype(np.uint8)


def tile_predict(model, x, tile_size, stride, blending, padding=True):
    h, w, c = x.shape
    ph = int(np.ceil(h / stride) * stride) if padding else h
    pw = int(np.ceil(w / stride) * stride) if padding else w
    canvas = np.zeros((ph, pw, c), np.float32)
    canvas[:h, :w] = x
    prob = np.zeros((ph, pw, 1), np.float32)
    wsum = np.zeros((ph, pw, 1), np.float32)
    win = blend_window(tile_size, blending)
    for y0 in range(0, ph - tile_size + 1, stride):
        for x0 in range(0, pw - tile_size + 1, stride):
            pred = model.predict(canvas[y0 : y0 + tile_size, x0 : x0 + tile_size][None, ...], verbose=0)[0]
            prob[y0 : y0 + tile_size, x0 : x0 + tile_size] += pred * win
            wsum[y0 : y0 + tile_size, x0 : x0 + tile_size] += win
    return (prob / np.maximum(wsum, 1e-7))[:h, :w]


def postprocess(mask, pp, hec=None):
    m = mask.astype(bool)
    if pp.get("opening", 0) > 0:
        m = scipy.ndimage.binary_opening(m, iterations=int(pp["opening"]))
    if pp.get("closing", 0) > 0:
        m = scipy.ndimage.binary_closing(m, iterations=int(pp["closing"]))
    if pp.get("fill_holes", 0) > 0:
        m = scipy.ndimage.binary_fill_holes(m)
    if pp.get("remove_small_components", 0) > 0:
        lab, n = scipy.ndimage.label(m)
        keep = np.zeros_like(m)
        for i in range(1, n + 1):
            c = lab == i
            if c.sum() >= int(pp["remove_small_components"]):
                keep |= c
        m = keep
    if pp.get("hec_guided", False) and hec is not None:
        m = m & (hec[..., 0] > 0)
    return m.astype(np.uint8)


def scores(y, p):
    inter = np.logical_and(y == 1, p == 1).sum()
    union = np.logical_or(y == 1, p == 1).sum()
    iou = (inter + 1e-7) / (union + 1e-7)
    dice = (2 * inter + 1e-7) / ((y == 1).sum() + (p == 1).sum() + 1e-7)
    return float(iou), float(dice)


def save_fig(fig_dir, sample_id, x, hec, y, p):
    rgb = x[..., :3]
    err = np.zeros((*y.shape[:2], 3), np.uint8)
    yy, pp = y[..., 0] > 0, p[..., 0] > 0
    err[np.logical_and(pp, ~yy)] = [255, 0, 0]
    err[np.logical_and(~pp, yy)] = [0, 0, 255]
    fig, ax = plt.subplots(1, 5, figsize=(14, 3))
    ax[0].imshow(rgb); ax[0].set_title("Planet RGB")
    ax[1].imshow(hec[..., 0], cmap="viridis"); ax[1].set_title("HEC-RAS")
    ax[2].imshow(y[..., 0], cmap="gray"); ax[2].set_title("QMask")
    ax[3].imshow(p[..., 0], cmap="gray"); ax[3].set_title("Prediction")
    ax[4].imshow(err); ax[4].set_title("Error FP/FN")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(fig_dir / f"{sample_id}.png", dpi=120); plt.close(fig)


def run(args):
    t0 = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(cfg["base_experiment_config"]).read_text())
    paths = yaml.safe_load(Path(base["paths_config"]).read_text())

    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    status = out / "status.txt"
    failed_csv = out / "failed_runs.csv"
    folds_csv = out / "full_image_validation_folds.csv"
    lb_csv = out / "full_image_validation_leaderboard.csv"
    test_summary_csv = out / "full_image_test_summary.csv"
    test_per_csv = out / "full_image_test_per_sample.csv"

    dcfg = DataConfig(manifest_path=paths["clean_manifest"], split_path=paths["split_file"], **base["data"])
    _, dev_df, test_df = load_manifest_and_split(dcfg)
    model = tf.keras.models.load_model(cfg["checkpoint_path"], custom_objects={"bce_dice_loss": bce_dice_loss, "iou_metric": iou_metric, "dice_coef": dice_coef}, compile=False)

    val_cfgs = cfg["validation"]["configs"]
    nfold = cfg["validation"]["n_splits"]
    gkf = GroupKFold(n_splits=nfold)
    ths = np.round(np.arange(cfg["validation"]["threshold_start"], cfg["validation"]["threshold_end"] + 1e-9, cfg["validation"]["threshold_step"]), 2)

    print(f"[{now()}] START total_configs={len(val_cfgs)} folds={nfold} dev_uuids={dev_df['UUID'].nunique()} test_uuids={test_df['UUID'].nunique()} dev_samples={len(dev_df)} test_samples={len(test_df)} output_dir={out} start_time={now()}")
    write_status(status, "start", elapsed=time.time() - t0)

    if args.dry_run:
        s = dev_df.iloc[0]
        x, y, _ = preprocess_full(s, dcfg)
        _ = tile_predict(model, x, 64, 64, "uniform", True)
        print(f"[{now()}] DRY-RUN complete x={x.shape} y={y.shape}")
        return

    folds_df = pd.read_csv(folds_csv) if folds_csv.exists() and folds_csv.stat().st_size > 0 else pd.DataFrame(columns=["config_id", "fold"])
    fold_rows = folds_df.to_dict("records") if len(folds_df) else []

    total_fold_tasks = len(val_cfgs) * (1 if args.quick_test else nfold)
    done_folds = 0

    for i, c in enumerate(val_cfgs, start=1):
        if args.quick_test and i > 1:
            break
        cfg_id = f"cfg_{i:02d}"
        elapsed = time.time() - t0
        print(f"[{now()}] CONFIG {i}/{len(val_cfgs)} id={cfg_id} tile_size={c['tile_size']} stride={c['stride']} blending={c['blending']} th_range={ths[0]}-{ths[-1]} post={c['postprocess']} now={now()} elapsed={elapsed:.1f}s")
        write_status(status, "config_start", cfg_id, elapsed=elapsed)

        splits = list(gkf.split(dev_df, groups=dev_df["UUID"]))
        if args.quick_test:
            splits = splits[:1]

        for fi, (tr, va) in enumerate(splits, start=1):
            current_folds_df = pd.read_csv(folds_csv) if folds_csv.exists() and folds_csv.stat().st_size > 0 else pd.DataFrame(columns=["config_id", "fold"])
            done = len(current_folds_df) and ((current_folds_df["config_id"] == cfg_id) & (current_folds_df["fold"] == fi)).any()
            if done and not args.rerun:
                done_folds += 1
                continue

            val_rows = dev_df.iloc[va].reset_index(drop=True)
            print(f"[{now()}] FOLD config_id={cfg_id} fold={fi}/{len(splits)} train_uuid={dev_df.iloc[tr]['UUID'].nunique()} val_uuid={val_rows['UUID'].nunique()} val_samples={len(val_rows)} now={now()}")
            tfold = time.time()
            write_status(status, "fold_start", cfg_id, fi, 0, time.time() - t0)
            try:
                n = len(val_rows) if not args.quick_test else min(20, len(val_rows))
                per_thr = {float(t): [] for t in ths}
                for k in range(n):
                    x, y, hec = preprocess_full(val_rows.iloc[k], dcfg)
                    prob = tile_predict(model, x, c["tile_size"], c["stride"], c["blending"], c["padding"])
                    for t in ths:
                        pp = postprocess((prob > t).astype(np.uint8), c["postprocess"], hec)
                        iou, dice = scores(y[..., 0], pp[..., 0])
                        per_thr[float(t)].append((iou, dice))
                    if (k + 1) % 10 == 0:
                        ee = time.time() - tfold
                        eta = (ee / (k + 1)) * (n - (k + 1))
                        print(f"[{now()}] val-progress {k+1}/{n} elapsed={ee:.1f}s eta={eta:.1f}s")
                        write_status(status, "fold_infer", cfg_id, fi, k + 1, time.time() - t0)

                best_iou, best_dice, best_t = -1, -1, 0.5
                for t, vals in per_thr.items():
                    mi = float(np.mean([v[0] for v in vals])); md = float(np.mean([v[1] for v in vals]))
                    if md > best_dice: best_iou, best_dice, best_t = mi, md, float(t)

                row = {"config_id": cfg_id, "fold": fi, "tile_size": c["tile_size"], "stride": c["stride"], "blending": c["blending"], "threshold": best_t, "postprocess": str(c["postprocess"]), "val_iou": best_iou, "val_dice": best_dice}
                fold_rows = [r for r in fold_rows if not (r.get("config_id") == cfg_id and int(r.get("fold", -1)) == fi)] + [row]
                pd.DataFrame(fold_rows).sort_values(["config_id", "fold"]).to_csv(folds_csv, index=False)

                done_folds += 1
                fold_elapsed = time.time() - tfold
                avg_fold = (time.time() - t0) / max(done_folds, 1)
                rem = max(total_fold_tasks - done_folds, 0) * avg_fold
                print(f"[{now()}] FOLD-DONE val_iou={best_iou:.6f} val_dice={best_dice:.6f} best_th={best_t:.2f} fold_elapsed={fold_elapsed:.1f}s eta_remaining={rem:.1f}s saved={folds_csv}")
            except Exception as e:
                pd.DataFrame([{"config_id": cfg_id, "fold": fi, "error": str(e), "time": now()}]).to_csv(failed_csv, mode="a", index=False, header=not failed_csv.exists())

        cfg_rows = [r for r in fold_rows if r["config_id"] == cfg_id]
        if len(cfg_rows) == len(splits):
            arr_i = np.array([r["val_iou"] for r in cfg_rows], dtype=float)
            arr_d = np.array([r["val_dice"] for r in cfg_rows], dtype=float)
            avg_t = float(np.mean([r["threshold"] for r in cfg_rows]))
            lb_row = {
                "config_id": cfg_id, "tile_size": c["tile_size"], "stride": c["stride"], "blending": c["blending"],
                "threshold": avg_t, "postprocess": str(c["postprocess"]),
                "mean_val_iou": float(arr_i.mean()), "std_val_iou": float(arr_i.std(ddof=0)),
                "mean_val_dice": float(arr_d.mean()), "std_val_dice": float(arr_d.std(ddof=0)),
            }
            lb = pd.read_csv(lb_csv) if lb_csv.exists() and lb_csv.stat().st_size > 0 else pd.DataFrame()
            lb = lb[lb["config_id"] != cfg_id] if len(lb) else lb
            lb = pd.concat([lb, pd.DataFrame([lb_row])], ignore_index=True).sort_values("mean_val_dice", ascending=False)
            lb.to_csv(lb_csv, index=False)
            best_now = lb.iloc[0]["config_id"] == cfg_id
            avg_cfg = (time.time() - t0) / i
            cfg_eta = max(len(val_cfgs) - i, 0) * avg_cfg
            print(f"[{now()}] CONFIG-DONE mean_val_iou={lb_row['mean_val_iou']:.6f} mean_val_dice={lb_row['mean_val_dice']:.6f} std_iou={lb_row['std_val_iou']:.6f} std_dice={lb_row['std_val_dice']:.6f} current_best={best_now} elapsed={time.time()-t0:.1f}s eta={cfg_eta:.1f}s")

    lb = pd.read_csv(lb_csv)
    best = lb.iloc[0].to_dict()
    print(f"[{now()}] BEST-VAL-CONFIG {best}")

    best_cfg = val_cfgs[int(best["config_id"].split("_")[1]) - 1]
    test_n = len(test_df) if not args.quick_test else min(20, len(test_df))
    th = float(best["threshold"])
    test_rows = []
    ttest = time.time()

    for k in range(test_n):
        x, y, hec = preprocess_full(test_df.iloc[k], dcfg)
        prob = tile_predict(model, x, best_cfg["tile_size"], best_cfg["stride"], best_cfg["blending"], best_cfg["padding"])
        pred = postprocess((prob > th).astype(np.uint8), best_cfg["postprocess"], hec)
        iou, dice = scores(y[..., 0], pred[..., 0])
        test_rows.append({"UUID": test_df.iloc[k]["UUID"], "date": test_df.iloc[k]["date"], "iou": iou, "dice": dice})

        if k < 3:
            save_fig(fig_dir, f"sample_{k:03d}", x, hec, y, pred)

        if (k + 1) % 10 == 0:
            ee = time.time() - ttest
            eta = (ee / (k + 1)) * (test_n - (k + 1))
            print(f"[{now()}] test-progress {k+1}/{test_n} elapsed={ee:.1f}s eta={eta:.1f}s")
            write_status(status, "test_infer", best["config_id"], "", k + 1, time.time() - t0)

    per_df = pd.DataFrame(test_rows)
    per_df.to_csv(test_per_csv, index=False)
    summary = pd.DataFrame([
        {"metric": "mean_iou", "value": float(per_df["iou"].mean())},
        {"metric": "std_iou", "value": float(per_df["iou"].std(ddof=0))},
        {"metric": "mean_dice", "value": float(per_df["dice"].mean())},
        {"metric": "std_dice", "value": float(per_df["dice"].std(ddof=0))},
        {"metric": "n_samples", "value": int(len(per_df))},
    ])
    summary.to_csv(test_summary_csv, index=False)

    per_df.groupby("UUID").agg(mean_iou=("iou", "mean"), mean_dice=("dice", "mean"), n=("UUID", "size")).reset_index().sort_values("mean_iou").to_csv(out / "error_analysis_by_uuid.csv", index=False)
    per_df.groupby("date").agg(mean_iou=("iou", "mean"), mean_dice=("dice", "mean"), n=("date", "size")).reset_index().sort_values("mean_iou").to_csv(out / "error_analysis_by_date.csv", index=False)
    per_df.sort_values("iou").head(20).to_csv(out / "worst_20_samples.csv", index=False)
    per_df.sort_values("iou", ascending=False).head(20).to_csv(out / "best_20_samples.csv", index=False)

    print(f"[{now()}] FINAL best_full_image_config={best['config_id']} final_heldout_iou={per_df['iou'].mean():.6f} final_heldout_dice={per_df['dice'].mean():.6f}")
    print(f"[{now()}] OUTPUTS: {lb_csv}, {test_summary_csv}, {test_per_csv}, {out/'error_analysis_by_uuid.csv'}, {out/'error_analysis_by_date.csv'}, {out/'worst_20_samples.csv'}, {out/'best_20_samples.csv'}, {failed_csv}, {status}, {fig_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/paper3_full_image.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick-test", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    run(ap.parse_args())
