import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import GroupKFold

from data import DataConfig, load_manifest_and_split, make_dataset
from losses import bce_dice_loss, dice_coef, iou_metric
from models import build_unet_binary


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    dcfg = DataConfig(manifest_path=paths["clean_manifest"], split_path=paths["split_file"], **cfg["data"])

    _, dev_df, test_df = load_manifest_and_split(dcfg)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    gkf = GroupKFold(n_splits=cfg["training"]["n_splits"])
    fold_rows = []
    for fold, (tr, va) in enumerate(gkf.split(dev_df, groups=dev_df["UUID"]), start=1):
        tr_df, va_df = dev_df.iloc[tr], dev_df.iloc[va]
        model = build_unet_binary(input_shape=(dcfg.image_size, dcfg.image_size, 9), **cfg["model"])
        model.compile(optimizer=tf.keras.optimizers.Adam(cfg["training"]["lr"]), loss=bce_dice_loss, metrics=["accuracy", iou_metric, dice_coef])
        h = model.fit(make_dataset(tr_df, dcfg, True), validation_data=make_dataset(va_df, dcfg, False), epochs=cfg["training"]["epochs"], verbose=2)
        best = int(np.argmin(h.history["val_loss"]))
        fold_rows.append({"fold": fold, "best_epoch": best + 1, "val_iou": float(h.history["val_iou_metric"][best]), "val_dice": float(h.history["val_dice_coef"][best])})

    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    final_epochs = int(round(np.mean([r["best_epoch"] for r in fold_rows])))

    final_model = build_unet_binary(input_shape=(dcfg.image_size, dcfg.image_size, 9), **cfg["model"])
    final_model.compile(optimizer=tf.keras.optimizers.Adam(cfg["training"]["lr"]), loss=bce_dice_loss, metrics=["accuracy", iou_metric, dice_coef])
    final_model.fit(make_dataset(dev_df, dcfg, True), epochs=final_epochs, verbose=2)
    eval_vals = final_model.evaluate(make_dataset(test_df, dcfg, False), verbose=0)
    names = final_model.metrics_names
    metrics = {k: float(v) for k, v in zip(names, eval_vals)}
    (out / "heldout_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    final_model.save(out / "unet_earlyfusion_final.keras")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/unet_earlyfusion.yaml")
    main(ap.parse_args().config)
