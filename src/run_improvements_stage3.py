import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import GroupKFold

from data import DataConfig, load_manifest_and_split, make_dataset, preprocess_sample
from evaluate_tiled_full_image import run_tiled_evaluation
from models import build_unet_binary
from run_experiments import focal_dice_loss, eval_threshold_metrics


def focal_tversky_loss(y_true, y_pred, gamma=0.75, alpha=0.7, beta=0.3):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    t = (tp + 1e-7) / (tp + alpha * fp + beta * fn + 1e-7)
    return tf.pow((1.0 - t), gamma)


def get_loss(name):
    return focal_dice_loss if name == 'focal_dice' else focal_tversky_loss


def smoke_test_128(df, cfg, out_dir):
    cfg128 = DataConfig(**{**cfg.__dict__, 'image_size': 128})
    idxs = random.sample(range(len(df)), k=min(50, len(df)))
    rows = []
    for i in idxs:
        r = df.iloc[i]
        x, y = preprocess_sample(r['planet_path'].encode(), r['hecras_path'].encode(), r['qmask_path'].encode(), cfg128, training=False)
        rows.append({'idx': i, 'x_shape': str(x.shape), 'y_shape': str(y.shape), 'ok': x.shape == (128, 128, 9) and y.shape == (128, 128, 1)})
    out = Path(out_dir) / 'smoke_test_128_shapes.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main(config_path='configs/experiments/improvement_stage_3.yaml'):
    cfg = yaml.safe_load(Path(config_path).read_text())
    base = yaml.safe_load(Path(cfg['base_experiment_config']).read_text())
    paths = yaml.safe_load(Path(base['paths_config']).read_text())
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)

    dcfg = DataConfig(manifest_path=paths['clean_manifest'], split_path=paths['split_file'], **base['data'])
    _, dev_df, test_df = load_manifest_and_split(dcfg)
    smoke_csv = smoke_test_128(dev_df, dcfg, out)

    fold_path, lb_path, final_path = out / 'fold_results.csv', out / 'leaderboard_validation.csv', out / 'final_test_results.csv'
    tiled_path, failed_path = out / 'tiled_full_image_test_results.csv', out / 'failed_runs.csv'
    notes_path = out / 'README_stage3.txt'

    gkf = GroupKFold(n_splits=base['training']['n_splits'])
    thresholds = cfg['thresholds']
    fold_rows, exp_rows = [], []

    for exp_id, cand in enumerate(cfg['candidates'], start=1):
        try:
            fold_scores = []
            for fold, (tr, va) in enumerate(gkf.split(dev_df, groups=dev_df['UUID']), start=1):
                t0 = time.time()
                tr_df, va_df = dev_df.iloc[tr], dev_df.iloc[va]
                tr_df = pd.concat([tr_df] * int(cand['tiles_per_image']), ignore_index=True)
                model = build_unet_binary(input_shape=(64, 64, 9), variant=cand['model_variant'], **base['model'])
                model.compile(optimizer=tf.keras.optimizers.Adam(base['training']['lr']), loss=get_loss(cand['loss']))
                model.fit(make_dataset(tr_df, dcfg, True), validation_data=make_dataset(va_df, dcfg, False), epochs=int(cand['epochs']), verbose=0)
                bt, bd, bi = eval_threshold_metrics(model, va_df, dcfg, thresholds)
                fold_rows.append({'exp_id': exp_id, 'fold': fold, **cand, 'val_dice': bd, 'val_iou': bi, 'threshold': bt, 'elapsed_sec': round(time.time() - t0, 1)})
                pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
                fold_scores.append((bd, bt))
            exp_rows.append({'exp_id': exp_id, **cand, 'mean_val_dice': float(np.mean([x[0] for x in fold_scores])), 'std_val_dice': float(np.std([x[0] for x in fold_scores])), 'mean_threshold': float(np.mean([x[1] for x in fold_scores]))})
            pd.DataFrame(exp_rows).sort_values('mean_val_dice', ascending=False).to_csv(lb_path, index=False)
        except Exception as e:
            pd.DataFrame([{'exp_id': exp_id, **cand, 'error': str(e)}]).to_csv(failed_path, mode='a', index=False, header=not failed_path.exists())

    lb = pd.read_csv(lb_path)
    best = lb.iloc[0].to_dict()
    model = build_unet_binary(input_shape=(64, 64, 9), variant=best['model_variant'], **base['model'])
    model.compile(optimizer=tf.keras.optimizers.Adam(base['training']['lr']), loss=get_loss(best['loss']))
    model.fit(make_dataset(dev_df, dcfg, True), epochs=int(best['epochs']), verbose=0)

    y_true, y_prob = [], []
    for xb, yb in make_dataset(test_df, dcfg, False):
        y_true.append(yb.numpy()); y_prob.append(model.predict(xb, verbose=0))
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    yp = (y_prob > float(best['mean_threshold'])).astype(np.float32)
    inter = np.sum(y_true * yp); union = np.sum(y_true) + np.sum(yp) - inter
    dice = (2 * inter + 1e-7) / (np.sum(y_true) + np.sum(yp) + 1e-7); iou = (inter + 1e-7) / (union + 1e-7)
    pd.DataFrame([{'stage': 'stage3_best', **best, 'test_iou': float(iou), 'test_dice': float(dice)}]).to_csv(final_path, index=False)

    # Ensemble with best stage2 SE by validation-selected threshold
    stage2_model_path = Path('results/improvement_stage_2/best_stage2_model.keras')
    if stage2_model_path.exists():
        stage2_model = tf.keras.models.load_model(stage2_model_path, compile=False)
        ens_true, ens_prob = [], []
        for xb, yb in make_dataset(test_df, dcfg, False):
            p1 = model.predict(xb, verbose=0)
            p2 = stage2_model.predict(xb, verbose=0)
            ens_true.append(yb.numpy()); ens_prob.append((p1 + p2) / 2.0)
        ens_true, ens_prob = np.concatenate(ens_true), np.concatenate(ens_prob)
        yp = (ens_prob > float(best['mean_threshold'])).astype(np.float32)
        inter = np.sum(ens_true * yp); union = np.sum(ens_true) + np.sum(yp) - inter
        dice2 = (2 * inter + 1e-7) / (np.sum(ens_true) + np.sum(yp) + 1e-7); iou2 = (inter + 1e-7) / (union + 1e-7)
        pd.concat([pd.read_csv(final_path), pd.DataFrame([{'stage': 'stage2_stage3_ensemble', **best, 'test_iou': float(iou2), 'test_dice': float(dice2)}])], ignore_index=True).to_csv(final_path, index=False)

    run_tiled_evaluation(model, test_df, dcfg, threshold=float(best['mean_threshold']), out_csv=str(tiled_path))

    notes_path.write_text(
        "Stage 3 ran only new candidates: se_unet tile4/tile6, se_unet focal_tversky tile4, cbam_unet tile4, dual_encoder_unet tile4.\n"
        "Selection metric: mean GroupKFold validation Dice. Held-out used only after selection.\n"
        f"128 smoke test file: {smoke_csv}\n"
    )


if __name__ == '__main__':
    main()
