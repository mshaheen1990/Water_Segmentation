import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import GroupKFold

from data import DataConfig, load_manifest_and_split, make_dataset
from models import build_unet_binary
from run_experiments import focal_dice_loss, eval_threshold_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    out = Path('results/reproducibility_se_unet')
    out.mkdir(parents=True, exist_ok=True)

    exp_cfg = yaml.safe_load(Path('configs/experiments/unet_earlyfusion.yaml').read_text())
    paths = yaml.safe_load(Path(exp_cfg['paths_config']).read_text())
    dcfg = DataConfig(manifest_path=paths['clean_manifest'], split_path=paths['split_file'], **exp_cfg['data'])
    _, dev_df, test_df = load_manifest_and_split(dcfg)

    seeds = [42, 1337, 2026]
    n_splits = exp_cfg['training']['n_splits']
    fixed_threshold = 0.552
    epochs = 20

    fold_rows = []
    final_rows = []

    for seed in seeds:
        set_seed(seed)
        gkf = GroupKFold(n_splits=n_splits)
        val_scores = []

        for fold, (tr, va) in enumerate(gkf.split(dev_df, groups=dev_df['UUID']), start=1):
            tr_df, va_df = dev_df.iloc[tr], dev_df.iloc[va]
            model = build_unet_binary(input_shape=(64, 64, 9), variant='se_unet', **exp_cfg['model'])
            model.compile(optimizer=tf.keras.optimizers.Adam(exp_cfg['training']['lr']), loss=focal_dice_loss)
            model.fit(make_dataset(tr_df, dcfg, True), validation_data=make_dataset(va_df, dcfg, False), epochs=epochs, verbose=0)
            _, val_dice, val_iou = eval_threshold_metrics(model, va_df, dcfg, [fixed_threshold])
            fold_rows.append({'seed': seed, 'fold': fold, 'val_dice': val_dice, 'val_iou': val_iou, 'threshold': fixed_threshold})
            val_scores.append(val_dice)
            pd.DataFrame(fold_rows).to_csv(out / 'fold_results.csv', index=False)

        # final retrain on dev and held-out evaluate
        set_seed(seed)
        model = build_unet_binary(input_shape=(64, 64, 9), variant='se_unet', **exp_cfg['model'])
        model.compile(optimizer=tf.keras.optimizers.Adam(exp_cfg['training']['lr']), loss=focal_dice_loss)
        model.fit(make_dataset(dev_df, dcfg, True), epochs=epochs, verbose=0)

        y_true, y_prob = [], []
        for xb, yb in make_dataset(test_df, dcfg, False):
            y_true.append(yb.numpy()); y_prob.append(model.predict(xb, verbose=0))
        y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
        yp = (y_prob > fixed_threshold).astype(np.float32)
        inter = np.sum(y_true * yp); union = np.sum(y_true) + np.sum(yp) - inter
        test_dice = (2 * inter + 1e-7) / (np.sum(y_true) + np.sum(yp) + 1e-7)
        test_iou = (inter + 1e-7) / (union + 1e-7)
        final_rows.append({'seed': seed, 'mean_val_dice': float(np.mean(val_scores)), 'test_iou': float(test_iou), 'test_dice': float(test_dice)})
        pd.DataFrame(final_rows).to_csv(out / 'final_test_results.csv', index=False)

    final_df = pd.DataFrame(final_rows)
    summary = pd.DataFrame([
        {'metric': 'validation_dice', 'mean': final_df['mean_val_dice'].mean(), 'std': final_df['mean_val_dice'].std(ddof=0)},
        {'metric': 'test_iou', 'mean': final_df['test_iou'].mean(), 'std': final_df['test_iou'].std(ddof=0)},
        {'metric': 'test_dice', 'mean': final_df['test_dice'].mean(), 'std': final_df['test_dice'].std(ddof=0)},
    ])
    summary.to_csv(out / 'summary_mean_std.csv', index=False)
    print(summary)


if __name__ == '__main__':
    main()
