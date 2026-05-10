import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import GroupKFold

from data import DataConfig, load_manifest_and_split, make_dataset
from evaluate_tiled_full_image import run_tiled_evaluation
from run_experiments import focal_dice_loss, eval_threshold_metrics
from models import build_unet_binary


def main(config_path='configs/experiments/improvement_stage_2.yaml'):
    cfg = yaml.safe_load(Path(config_path).read_text())
    base = yaml.safe_load(Path(cfg['base_experiment_config']).read_text())
    paths = yaml.safe_load(Path(base['paths_config']).read_text())
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)

    dcfg = DataConfig(manifest_path=paths['clean_manifest'], split_path=paths['split_file'], **base['data'])
    _, dev_df, test_df = load_manifest_and_split(dcfg)

    candidates = cfg['candidates']
    n_splits = base['training']['n_splits']
    gkf = GroupKFold(n_splits=n_splits)
    thresholds = cfg['thresholds']

    fold_path = out / 'fold_results.csv'
    lb_path = out / 'leaderboard_validation.csv'
    final_path = out / 'final_test_results.csv'
    tiled_path = out / 'tiled_full_image_test_results.csv'

    fold_rows, exp_rows = [], []
    for exp_id, cand in enumerate(candidates, start=1):
        variant = cand['model_variant']
        epochs = cand['epochs']
        tile_repeats = cand.get('tiles_per_image', 1)
        print(f"[STAGE2] exp {exp_id}/{len(candidates)} variant={variant} epochs={epochs} tiles_per_image={tile_repeats}")

        fold_scores = []
        for fold, (tr, va) in enumerate(gkf.split(dev_df, groups=dev_df['UUID']), start=1):
            t0 = time.time()
            tr_df, va_df = dev_df.iloc[tr], dev_df.iloc[va]
            tr_df = pd.concat([tr_df] * tile_repeats, ignore_index=True)
            model = build_unet_binary(input_shape=(dcfg.image_size, dcfg.image_size, 9), variant=variant, **base['model'])
            model.compile(optimizer=tf.keras.optimizers.Adam(base['training']['lr']), loss=focal_dice_loss, metrics=[])
            model.fit(make_dataset(tr_df, dcfg, True), validation_data=make_dataset(va_df, dcfg, False), epochs=epochs, verbose=0)
            bt, bd, bi = eval_threshold_metrics(model, va_df, dcfg, thresholds)
            fold_rows.append({'exp_id': exp_id, 'fold': fold, 'model_variant': variant, 'epochs': epochs, 'val_dice': bd, 'val_iou': bi, 'threshold': bt, 'elapsed_sec': round(time.time()-t0,1)})
            pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
            fold_scores.append((bd, bt))

        mean_d = float(np.mean([x[0] for x in fold_scores]))
        std_d = float(np.std([x[0] for x in fold_scores]))
        mean_t = float(np.mean([x[1] for x in fold_scores]))
        exp_rows.append({'exp_id': exp_id, 'model_variant': variant, 'epochs': epochs, 'mean_val_dice': mean_d, 'std_val_dice': std_d, 'mean_threshold': mean_t})
        pd.DataFrame(exp_rows).sort_values('mean_val_dice', ascending=False).to_csv(lb_path, index=False)

    best = pd.DataFrame(exp_rows).sort_values('mean_val_dice', ascending=False).iloc[0]
    model = build_unet_binary(input_shape=(dcfg.image_size, dcfg.image_size, 9), variant=best['model_variant'], **base['model'])
    model.compile(optimizer=tf.keras.optimizers.Adam(base['training']['lr']), loss=focal_dice_loss)
    model.fit(make_dataset(dev_df, dcfg, True), epochs=int(best['epochs']), verbose=0)
    ckpt = out / 'best_stage2_model.keras'
    model.save(ckpt)

    y_true, y_prob = [], []
    for xb, yb in make_dataset(test_df, dcfg, False):
        y_true.append(yb.numpy()); y_prob.append(model.predict(xb, verbose=0))
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    yp = (y_prob > float(best['mean_threshold'])).astype(np.float32)
    inter = np.sum(y_true * yp); union = np.sum(y_true) + np.sum(yp) - inter
    dice = (2*inter+1e-7)/(np.sum(y_true)+np.sum(yp)+1e-7); iou = (inter+1e-7)/(union+1e-7)
    pd.DataFrame([{'model_variant': best['model_variant'], 'epochs': int(best['epochs']), 'threshold': float(best['mean_threshold']), 'test_iou': float(iou), 'test_dice': float(dice)}]).to_csv(final_path, index=False)

    run_tiled_evaluation(model, test_df, dcfg, threshold=float(best['mean_threshold']), out_csv=str(tiled_path))


if __name__ == '__main__':
    main()
