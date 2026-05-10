import argparse, json
from pathlib import Path

import tensorflow as tf
import yaml

from data import DataConfig, load_manifest_and_split, make_dataset
from losses import bce_dice_loss, dice_coef, iou_metric


def main(exp_cfg):
    cfg = yaml.safe_load(Path(exp_cfg).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    dcfg = DataConfig(manifest_path=paths["clean_manifest"], split_path=paths["split_file"], **cfg["data"])
    _, _, test_df = load_manifest_and_split(dcfg)

    model = tf.keras.models.load_model(cfg["eval"]["model_path"], custom_objects={"bce_dice_loss": bce_dice_loss, "iou_metric": iou_metric, "dice_coef": dice_coef})
    vals = model.evaluate(make_dataset(test_df, dcfg, False), verbose=0)
    metrics = {k: float(v) for k, v in zip(model.metrics_names, vals)}
    print(metrics)
    Path(cfg["eval"]["output_json"]).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/unet_earlyfusion.yaml")
    main(ap.parse_args().config)
