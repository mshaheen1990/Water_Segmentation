import argparse, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
import tensorflow as tf
import yaml
from sklearn.model_selection import GroupKFold

from data import DataConfig, load_manifest_and_split, _fix_channels, _load_tif, _minmax_norm
from losses import bce_dice_loss, dice_coef, iou_metric


def now(): return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')

def status_write(path, stage, cfg='', fold='', processed=0, elapsed=0):
    path.write_text(f"last_update={now()}\nstage={stage}\nconfig={cfg}\nfold={fold}\nprocessed={processed}\nelapsed={elapsed:.1f}\n")

def win(size, b):
    if b=='hann': h=np.hanning(size); w=np.outer(h,h); return (w/(w.max()+1e-7))[...,None].astype(np.float32)
    return np.ones((size,size,1),np.float32)

def preprocess(row, d):
    p=_fix_channels(_load_tif(row['planet_path']),d.planet_channels)
    h=_fix_channels(_load_tif(row['hecras_path']),d.hec_channels)
    q=_load_tif(row['qmask_path'])
    if h.shape[:2]!=p.shape[:2]: h=scipy.ndimage.zoom(h,(p.shape[0]/h.shape[0],p.shape[1]/h.shape[1],1),order=1)
    if q.shape[:2]!=p.shape[:2]: q=scipy.ndimage.zoom(q,(p.shape[0]/q.shape[0],p.shape[1]/q.shape[1],1),order=0)
    x=np.concatenate([_minmax_norm(p),(h>0).astype(np.float32)],-1); y=(q>0).astype(np.uint8)[...,:1]
    return x,y,(h>0).astype(np.uint8)

def tile_predict(model,x,tile,stride,blend):
    h,w,c=x.shape; ph=int(np.ceil(h/stride)*stride); pw=int(np.ceil(w/stride)*stride)
    canvas=np.zeros((ph,pw,c),np.float32); canvas[:h,:w]=x
    p=np.zeros((ph,pw,1),np.float32); s=np.zeros((ph,pw,1),np.float32); ww=win(tile,blend)
    for y0 in range(0,ph-tile+1,stride):
        for x0 in range(0,pw-tile+1,stride):
            pr=model.predict(canvas[y0:y0+tile,x0:x0+tile][None,...],verbose=0)[0]
            p[y0:y0+tile,x0:x0+tile]+=pr*ww; s[y0:y0+tile,x0:x0+tile]+=ww
    return (p/np.maximum(s,1e-7))[:h,:w]

def post(m,pp):
    b=m.astype(bool)
    if pp.get('opening',0)>0: b=scipy.ndimage.binary_opening(b,iterations=int(pp['opening']))
    if pp.get('closing',0)>0: b=scipy.ndimage.binary_closing(b,iterations=int(pp['closing']))
    if pp.get('fill_holes',0)>0: b=scipy.ndimage.binary_fill_holes(b)
    if pp.get('remove_small_components',0)>0:
        lab,n=scipy.ndimage.label(b); keep=np.zeros_like(b)
        for i in range(1,n+1):
            c=lab==i
            if c.sum()>=int(pp['remove_small_components']): keep|=c
        b=keep
    return b.astype(np.uint8)

def metrics(y,p):
    inter=np.logical_and(y==1,p==1).sum(); union=np.logical_or(y==1,p==1).sum()
    iou=(inter+1e-7)/(union+1e-7); dice=(2*inter+1e-7)/((y==1).sum()+(p==1).sum()+1e-7)
    return float(iou),float(dice)

def main(args):
    t0=time.time(); cfg=yaml.safe_load(Path(args.config).read_text()); base=yaml.safe_load(Path(cfg['base_experiment_config']).read_text()); paths=yaml.safe_load(Path(base['paths_config']).read_text())
    out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    status=out/'status.txt'; failed=out/'failed_runs.csv'; folds_csv=out/'validation_folds.csv'; lb_csv=out/'validation_leaderboard.csv'
    test_sum=out/'test_summary.csv'; test_per=out/'test_per_sample.csv'; dbg_csv=out/'postprocess_debug_per_sample.csv'; zero_csv=out/'zero_score_cases.csv'; empty_csv=out/'empty_prediction_cases.csv'

    d=DataConfig(manifest_path=paths['clean_manifest'], split_path=paths['split_file'], **base['data'])
    _,dev,test=load_manifest_and_split(d)
    model=tf.keras.models.load_model(cfg['checkpoint_path'],custom_objects={'bce_dice_loss':bce_dice_loss,'iou_metric':iou_metric,'dice_coef':dice_coef},compile=False)
    ths=np.round(np.arange(cfg['validation']['threshold_start'],cfg['validation']['threshold_end']+1e-9,cfg['validation']['threshold_step']),2)
    gkf=GroupKFold(n_splits=cfg['validation']['n_splits'])

    configs=cfg['validation']['configs']
    if args.quick_test: configs=[c for c in configs if c['config_id'] in ['cfg_01','cfg_02','cfg_03']]
    print(f"[{now()}] total_configs={len(configs)} folds={cfg['validation']['n_splits']} out={out}")

    if args.dry_run:
        row=dev.iloc[0]
        for c in configs[:5]:
            x,y,_=preprocess(row,d); prob=tile_predict(model,x,c['tile_size'],c['stride'],c['blending'])
            pb=(prob>0.552).astype(np.uint8); pa=post(pb,c['postprocess'])
            print(c['config_id'], 'shape_ok', prob.shape[:2]==y.shape[:2], 'before_pos', int(pb.sum()), 'after_pos', int(pa.sum()))
        return

    fold_rows=pd.read_csv(folds_csv).to_dict('records') if folds_csv.exists() and folds_csv.stat().st_size>0 else []
    debug_rows=[]

    for ci,c in enumerate(configs, start=1):
        print(f"[{now()}] CONFIG {ci}/{len(configs)} id={c['config_id']} tile={c['tile_size']} stride={c['stride']} blend={c['blending']} pp={c['postprocess']}")
        splits=list(gkf.split(dev,groups=dev['UUID']))
        if args.quick_test: splits=splits[:1]
        for fi,(tr,va) in enumerate(splits, start=1):
            done=any((r['config_id']==c['config_id'] and int(r['fold'])==fi) for r in fold_rows)
            if done and not args.rerun: continue
            val=dev.iloc[va].reset_index(drop=True)
            n=min(30,len(val)) if args.quick_test else len(val)
            print(f"[{now()}] FOLD {fi}/{len(splits)} val_samples={n}")
            fold_t=time.time(); best=(-1,-1,0.5); empty_count=0
            per_thr={float(t):[] for t in ths}
            for i in range(n):
                x,y,_=preprocess(val.iloc[i],d); prob=tile_predict(model,x,c['tile_size'],c['stride'],c['blending'])
                for t in ths:
                    pb=(prob>t).astype(np.uint8); pa=post(pb,c['postprocess'])
                    iob,dib=metrics(y[...,0],pb[...,0]); ioa,dia=metrics(y[...,0],pa[...,0])
                    before=int(pb.sum()); after=int(pa.sum()); gt=int(y.sum())
                    removed=(before-after)/max(before,1)
                    is_empty=after==0
                    if is_empty: empty_count+=1
                    debug_rows.append({'config_id':c['config_id'],'fold':fi,'sample_idx':i,'threshold':float(t),'pred_pos_before':before,'pred_pos_after':after,'gt_pos':gt,'pct_removed':removed,'became_empty':is_empty,'iou_before':iob,'dice_before':dib,'iou_after':ioa,'dice_after':dia})
                    per_thr[float(t)].append((ioa,dia))
                if (i+1)%10==0:
                    e=time.time()-fold_t; eta=(e/(i+1))*(n-(i+1)); print(f"[{now()}] processed {i+1}/{n} elapsed={e:.1f}s eta={eta:.1f}s")
                    status_write(status,'val_infer',c['config_id'],fi,i+1,time.time()-t0)
            pd.DataFrame(debug_rows).to_csv(dbg_csv,index=False)

            for t,v in per_thr.items():
                mi=np.mean([x[0] for x in v]); md=np.mean([x[1] for x in v])
                if md>best[1]: best=(mi,md,t)

            empty_rate=empty_count/max(n*len(ths),1)
            unsafe=empty_rate>0.20
            row={'config_id':c['config_id'],'fold':fi,'tile_size':c['tile_size'],'stride':c['stride'],'blending':c['blending'],'postprocess':str(c['postprocess']),'threshold':best[2],'val_iou':best[0],'val_dice':best[1],'empty_rate':empty_rate,'unsafe':unsafe}
            fold_rows=[r for r in fold_rows if not(r['config_id']==c['config_id'] and int(r['fold'])==fi)] + [row]
            pd.DataFrame(fold_rows).sort_values(['config_id','fold']).to_csv(folds_csv,index=False)
            print(f"[{now()}] FOLD-DONE iou={best[0]:.6f} dice={best[1]:.6f} th={best[2]:.2f} empty_rate={empty_rate:.3f} saved={folds_csv}")
            if unsafe:
                pd.DataFrame([{'config_id':c['config_id'],'fold':fi,'reason':'empty_rate>0.2','empty_rate':empty_rate,'time':now()}]).to_csv(failed,mode='a',index=False,header=not failed.exists())

        cr=[r for r in fold_rows if r['config_id']==c['config_id']]
        if len(cr)==len(splits):
            arr_i=np.array([r['val_iou'] for r in cr]); arr_d=np.array([r['val_dice'] for r in cr]); th=float(np.mean([r['threshold'] for r in cr]))
            unsafe=any(r.get('unsafe',False) for r in cr)
            lb=pd.read_csv(lb_csv) if lb_csv.exists() and lb_csv.stat().st_size>0 else pd.DataFrame()
            row={'config_id':c['config_id'],'tile_size':c['tile_size'],'stride':c['stride'],'blending':c['blending'],'postprocess':str(c['postprocess']),'threshold':th,'mean_val_iou':arr_i.mean(),'std_val_iou':arr_i.std(ddof=0),'mean_val_dice':arr_d.mean(),'std_val_dice':arr_d.std(ddof=0),'unsafe':unsafe}
            lb=lb[lb['config_id']!=c['config_id']] if len(lb) else lb
            lb=pd.concat([lb,pd.DataFrame([row])],ignore_index=True)
            safe_lb=lb[lb['unsafe']==False] if len(lb) else lb
            lb = pd.concat([safe_lb.sort_values('mean_val_dice',ascending=False), lb[lb['unsafe']==True]], ignore_index=True)
            lb.to_csv(lb_csv,index=False)
            print(f"[{now()}] CONFIG-DONE mean_iou={row['mean_val_iou']:.6f} mean_dice={row['mean_val_dice']:.6f} unsafe={unsafe} saved={lb_csv}")

    dbg=pd.read_csv(dbg_csv) if dbg_csv.exists() and dbg_csv.stat().st_size>0 else pd.DataFrame()
    if len(dbg):
        dbg[(dbg['iou_after']<1e-4)&(dbg['dice_after']<1e-4)].to_csv(zero_csv,index=False)
        dbg[dbg['became_empty']==True].to_csv(empty_csv,index=False)

    lb=pd.read_csv(lb_csv)
    lb=lb[lb['unsafe']==False]
    best=lb.iloc[0].to_dict()
    c=next(x for x in configs if x['config_id']==best['config_id'])
    th=float(best['threshold'])
    n=min(30,len(test)) if args.quick_test else len(test)
    rows=[]; tt=time.time()
    for i in range(n):
        x,y,_=preprocess(test.iloc[i],d); prob=tile_predict(model,x,c['tile_size'],c['stride'],c['blending']); pa=post((prob>th).astype(np.uint8),c['postprocess'])
        iou,dice=metrics(y[...,0],pa[...,0]); rows.append({'UUID':test.iloc[i]['UUID'],'date':test.iloc[i]['date'],'iou':iou,'dice':dice})
        if (i+1)%10==0:
            e=time.time()-tt; eta=(e/(i+1))*(n-(i+1)); print(f"[{now()}] test {i+1}/{n} elapsed={e:.1f}s eta={eta:.1f}s")
            status_write(status,'test_infer',best['config_id'],'',i+1,time.time()-t0)
    per=pd.DataFrame(rows); per.to_csv(test_per,index=False)
    pd.DataFrame([{'metric':'mean_iou','value':per['iou'].mean()},{'metric':'mean_dice','value':per['dice'].mean()},{'metric':'std_iou','value':per['iou'].std(ddof=0)},{'metric':'std_dice','value':per['dice'].std(ddof=0)}]).to_csv(test_sum,index=False)
    print(f"[{now()}] FINAL best={best['config_id']} iou={per['iou'].mean():.6f} dice={per['dice'].mean():.6f}")
    print(f"outputs: {folds_csv}, {lb_csv}, {test_sum}, {test_per}, {dbg_csv}, {zero_csv}, {empty_csv}, {failed}, {status}")

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/experiments/paper3_overlap_ablation.yaml'); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--quick-test',action='store_true'); ap.add_argument('--rerun',action='store_true')
    main(ap.parse_args())
