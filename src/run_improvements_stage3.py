import argparse
import random
import time
from datetime import datetime
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


def write_status(path, msg):
    path.write_text(f"[{datetime.utcnow().isoformat()}Z] {msg}\n")


def main(config_path='configs/experiments/improvement_stage_3.yaml', quick_test=False, rerun=False):
    cfg = yaml.safe_load(Path(config_path).read_text())
    base = yaml.safe_load(Path(cfg['base_experiment_config']).read_text())
    paths = yaml.safe_load(Path(base['paths_config']).read_text())
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)

    dcfg = DataConfig(manifest_path=paths['clean_manifest'], split_path=paths['split_file'], **base['data'])
    _, dev_df, test_df = load_manifest_and_split(dcfg)
    smoke_csv = smoke_test_128(dev_df, dcfg, out)

    fold_path, lb_path, final_path = out / 'fold_results.csv', out / 'leaderboard_validation.csv', out / 'final_test_results.csv'
    tiled_path, failed_path, status_path = out / 'tiled_full_image_test_results.csv', out / 'failed_runs.csv', out / 'status.txt'
    notes_path = out / 'README_stage3.txt'

    existing_folds = pd.read_csv(fold_path) if fold_path.exists() and fold_path.stat().st_size > 0 else pd.DataFrame(columns=['exp_id', 'fold'])
    fold_rows = existing_folds.to_dict('records') if len(existing_folds) else []
    exp_rows = pd.read_csv(lb_path).to_dict('records') if lb_path.exists() and lb_path.stat().st_size > 0 else []

    candidates = cfg['candidates'][:1] if quick_test else cfg['candidates']
    gkf = GroupKFold(n_splits=base['training']['n_splits'])
    thresholds = cfg['thresholds']

    write_status(status_path, f"Stage3 start quick_test={quick_test} candidates={len(candidates)}")

    for exp_idx, cand in enumerate(candidates, start=1):
        exp_id = exp_idx
        print(f"\n[EXP-START] {exp_idx}/{len(candidates)} exp_id={exp_id} cand={cand}")
        write_status(status_path, f"EXP start exp_id={exp_id} cand={cand}")

        fold_scores = []
        fold_splits = list(gkf.split(dev_df, groups=dev_df['UUID']))
        if quick_test:
            fold_splits = fold_splits[:1]

        for fold, (tr, va) in enumerate(fold_splits, start=1):
            done = len(existing_folds) and ((existing_folds['exp_id'] == exp_id) & (existing_folds['fold'] == fold)).any()
            if done and not rerun:
                print(f"[RESUME-SKIP] exp_id={exp_id} fold={fold}")
                continue

            tr_df, va_df = dev_df.iloc[tr], dev_df.iloc[va]
            tr_df = pd.concat([tr_df] * int(cand['tiles_per_image']), ignore_index=True)
            print(f"[FOLD-START] exp_id={exp_id} fold={fold}/{len(fold_splits)} train={len(tr_df)} val={len(va_df)}")
            write_status(status_path, f"FOLD start exp_id={exp_id} fold={fold} train={len(tr_df)} val={len(va_df)}")

            t0 = time.time()
            try:
                model = build_unet_binary(input_shape=(64, 64, 9), variant=cand['model_variant'], **base['model'])
                model.compile(optimizer=tf.keras.optimizers.Adam(base['training']['lr']), loss=get_loss(cand['loss']))

                epochs = 1 if quick_test else int(cand['epochs'])
                cb = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: write_status(
                        status_path,
                        f"epoch_end exp_id={exp_id} fold={fold} epoch={epoch+1}/{epochs} loss={logs.get('loss')} val_loss={logs.get('val_loss')}"
                    )
                )
                model.fit(make_dataset(tr_df, dcfg, True), validation_data=make_dataset(va_df, dcfg, False), epochs=epochs, verbose=1, callbacks=[cb])

                bt, bd, bi = eval_threshold_metrics(model, va_df, dcfg, thresholds)
                row = {'exp_id': exp_id, 'fold': fold, **cand, 'val_dice': bd, 'val_iou': bi, 'threshold': bt, 'elapsed_sec': round(time.time() - t0, 1)}
                fold_rows.append(row)
                pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
                existing_folds = pd.DataFrame(fold_rows)
                fold_scores.append((bd, bt))
                print(f"[FOLD-DONE] exp_id={exp_id} fold={fold} val_dice={bd:.6f} threshold={bt:.3f} saved={fold_path}")
                write_status(status_path, f"FOLD done exp_id={exp_id} fold={fold} val_dice={bd:.6f} threshold={bt:.3f}")
            except Exception as e:
                pd.DataFrame([{'exp_id': exp_id, 'fold': fold, **cand, 'error': str(e)}]).to_csv(failed_path, mode='a', index=False, header=not failed_path.exists())
                write_status(status_path, f"FOLD fail exp_id={exp_id} fold={fold} err={e}")

        exp_fold_df = pd.DataFrame([r for r in fold_rows if r['exp_id'] == exp_id])
        if len(exp_fold_df) == len(fold_splits):
            exp_row = {'exp_id': exp_id, **cand, 'mean_val_dice': float(exp_fold_df['val_dice'].mean()), 'std_val_dice': float(exp_fold_df['val_dice'].std(ddof=0)), 'mean_threshold': float(exp_fold_df['threshold'].mean())}
            exp_rows = [r for r in exp_rows if r.get('exp_id') != exp_id] + [exp_row]
            pd.DataFrame(exp_rows).sort_values('mean_val_dice', ascending=False).to_csv(lb_path, index=False)
            print(f"[EXP-DONE] exp_id={exp_id} mean_val_dice={exp_row['mean_val_dice']:.6f}")
            write_status(status_path, f"EXP done exp_id={exp_id} mean_val_dice={exp_row['mean_val_dice']:.6f}")

    if quick_test:
        write_status(status_path, "Quick test completed; skipping final held-out evaluation")
        return

    lb = pd.read_csv(lb_path)
    best = lb.iloc[0].to_dict()
    model = build_unet_binary(input_shape=(64, 64, 9), variant=best['model_variant'], **base['model'])
    model.compile(optimizer=tf.keras.optimizers.Adam(base['training']['lr']), loss=get_loss(best['loss']))
    model.fit(make_dataset(dev_df, dcfg, True), epochs=int(best['epochs']), verbose=1)

    y_true, y_prob = [], []
    for xb, yb in make_dataset(test_df, dcfg, False):
        y_true.append(yb.numpy()); y_prob.append(model.predict(xb, verbose=0))
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    yp = (y_prob > float(best['mean_threshold'])).astype(np.float32)
    inter = np.sum(y_true * yp); union = np.sum(y_true) + np.sum(yp) - inter
    dice = (2 * inter + 1e-7) / (np.sum(y_true) + np.sum(yp) + 1e-7); iou = (inter + 1e-7) / (union + 1e-7)
    pd.DataFrame([{'stage': 'stage3_best', **best, 'test_iou': float(iou), 'test_dice': float(dice)}]).to_csv(final_path, index=False)
    run_tiled_evaluation(model, test_df, dcfg, threshold=float(best['mean_threshold']), out_csv=str(tiled_path))

    notes_path.write_text(
        "Stage 3 candidates executed with resume and heartbeat support.\n"
        f"quick_test=False run at {datetime.utcnow().isoformat()}Z\n"
        f"128 smoke test file: {smoke_csv}\n"
    )
    write_status(status_path, "Stage3 completed")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/experiments/improvement_stage_3.yaml')
    ap.add_argument('--quick_test', action='store_true')
    ap.add_argument('--rerun', action='store_true')
    args = ap.parse_args()
    main(config_path=args.config, quick_test=args.quick_test, rerun=args.rerun)
