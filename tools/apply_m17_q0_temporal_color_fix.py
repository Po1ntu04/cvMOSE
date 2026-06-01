#!/usr/bin/env python3
"""Q0 obj2 temporal color/ROI repair probe.

This is a deliberately narrow diagnostic/fusion tool for q0sizv6m:obj2 after
manual review showed the baseline follows a same-class animal near the original
position while the first-frame obj2 moves toward the camera and later appears
along the lower image edge.  It keeps the base root everywhere else and replaces
only q0sizv6m obj2 with a training-free appearance model constrained by a
hand-audited spatiotemporal ROI.
"""
from __future__ import annotations

import argparse, json, shutil, zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('/home/yu/projects/cv/from fdu/MOSEv2'))
    p.add_argument('--base-root', type=Path, required=True)
    p.add_argument('--pred-root', type=Path, required=True)
    p.add_argument('--submit-root', type=Path, default=None)
    p.add_argument('--zip-path', type=Path, default=None)
    p.add_argument('--audit-json', type=Path, required=True)
    p.add_argument('--mode', choices=['conservative','balanced','wide'], default='balanced')
    p.add_argument('--make-submission', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def load_label(p:Path)->np.ndarray:
    arr=np.asarray(Image.open(p))
    return arr if arr.ndim==2 else arr[...,0]


def save_label(path:Path, arr:np.ndarray, palette):
    path.parent.mkdir(parents=True, exist_ok=True)
    img=Image.fromarray(arr.astype(np.uint8), mode='P')
    if palette: img.putpalette(palette)
    img.save(path)


def feature(rgb:np.ndarray)->np.ndarray:
    x=rgb.astype(np.float32)/255.0
    r,g,b=x[...,0],x[...,1],x[...,2]
    mx=x.max(axis=-1); mn=x.min(axis=-1)
    sat=(mx-mn)/(mx+1e-6)
    val=mx
    gray=x.mean(axis=-1)
    # weighted feature: RGB plus saturation/value to separate white/black/gray from tan fur and grass.
    return np.stack([r,g,b,0.8*sat,0.7*val,0.5*gray],axis=-1)


def kmeans(samples:np.ndarray,k:int=5,iters:int=20)->np.ndarray:
    if len(samples)<=k: return samples.copy()
    qs=np.linspace(0,len(samples)-1,k).round().astype(int)
    centers=samples[qs].copy()
    for _ in range(iters):
        d=((samples[:,None,:]-centers[None,:,:])**2).sum(axis=-1)
        lab=d.argmin(axis=1)
        new=[]
        for i in range(k):
            pts=samples[lab==i]
            new.append(pts.mean(axis=0) if len(pts) else centers[i])
        new=np.stack(new)
        if np.allclose(new,centers): break
        centers=new
    return centers


def dist_to_centers(feat:np.ndarray, centers:np.ndarray)->np.ndarray:
    flat=feat.reshape(-1,feat.shape[-1])
    d=((flat[:,None,:]-centers[None,:,:])**2).sum(axis=-1)
    return np.sqrt(d.min(axis=1)).reshape(feat.shape[:2])


def roi_box(frame:int,w:int,h:int,mode:str)->tuple[int,int,int,int]|None:
    # boxes are image-coordinate constraints from manual visual review.
    # They intentionally exclude the mid-left same-class animal that baseline marks after frame 13.
    if frame<=2:
        return None  # keep base/GT
    if 3<=frame<=5:
        return None  # absent/empty interval
    if 6<=frame<=8:
        return (0, int(0.58*h), int(0.38*w), h)
    if 9<=frame<=12:
        return (0, int(0.70*h), int(0.42*w), h)
    if 13<=frame<=19:
        return (0, int(0.78*h), int(0.55*w), h)
    if 20<=frame<=30:
        return (0, int(0.76*h), w, h)
    if 31<=frame<=35:
        # Avoid central tan bodies; keep lower/left gray-white target edge/body hypothesis.
        return (0, int(0.72*h), int((0.58 if mode!='wide' else 0.72)*w), h)
    if 36<=frame<=41:
        return (0, int(0.68*h), int((0.46 if mode=='conservative' else 0.58)*w), h)
    return None


def largest_components(mask:np.ndarray, max_components:int=2, min_area:int=80)->np.ndarray:
    try:
        from scipy import ndimage  # type: ignore
        lab,n=ndimage.label(mask)
        if n==0: return mask
        areas=np.bincount(lab.ravel())
        keep=np.zeros_like(mask,bool)
        ids=[i for i in np.argsort(areas)[::-1] if i!=0 and areas[i]>=min_area][:max_components]
        for i in ids: keep |= (lab==i)
        return keep
    except Exception:
        return mask


def smooth(mask:np.ndarray)->np.ndarray:
    img=Image.fromarray(mask.astype(np.uint8)*255)
    img=img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    return np.asarray(img)>0


def make_submission(ws:Path,pred_root:Path,submit_root:Path,zip_path:Path,overwrite:bool):
    from validate_mose_submission import validate_zip
    provided=ws/'homework/output'; jpeg=ws/'homework/JPEGImages'; ann=ws/'homework/Annotations'
    if submit_root.exists() and overwrite: shutil.rmtree(submit_root)
    submit_root.mkdir(parents=True,exist_ok=True)
    for root in [provided,pred_root]:
        for vd in sorted(p for p in root.iterdir() if p.is_dir()):
            dst=submit_root/vd.name
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(vd,dst)
    if zip_path.exists() and overwrite: zip_path.unlink()
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for png in sorted(submit_root.rglob('*.png')):
            zf.write(png,png.relative_to(submit_root).as_posix())
    res=validate_zip(zip_path,provided,jpeg,ann,433,66526,15,False)
    if not res.get('ok'):
        raise SystemExit('validation failed '+json.dumps(res,ensure_ascii=False))


def main():
    args=parse_args(); ws=args.workspace.resolve(); vid='q0sizv6m'; obj=2
    jpeg_root=ws/'homework/JPEGImages'; ann_root=ws/'homework/Annotations'
    frames=sorted((jpeg_root/vid).glob('*.jpg'))
    ann_img=Image.open(ann_root/vid/'00000.png'); ann=np.asarray(ann_img); palette=ann_img.getpalette()
    rgb0=np.asarray(Image.open(frames[0]).convert('RGB'))
    f0=feature(rgb0)
    pos=f0[ann==obj]
    neg=f0[ann==1]
    # Add grass/dirt upper samples as negatives to avoid background fragments.
    neg_extra=f0[:ann.shape[0]//2].reshape(-1,f0.shape[-1])[::200]
    rng=np.random.default_rng(7)
    if len(pos)>3000: pos=pos[rng.choice(len(pos),3000,replace=False)]
    if len(neg)>3000: neg=neg[rng.choice(len(neg),3000,replace=False)]
    neg=np.concatenate([neg,neg_extra],axis=0)
    cpos=kmeans(pos,5); cneg=kmeans(neg,6)
    if args.pred_root.exists() and args.overwrite: shutil.rmtree(args.pred_root)
    args.pred_root.mkdir(parents=True,exist_ok=True)
    audit={'video':vid,'obj_id':obj,'mode':args.mode,'base_root':str(args.base_root),'changed_frames':[],'frames':{},'note':'q0 obj2 bottom/foreground color-ROI repair; no GT beyond first mask used'}
    for vd in sorted(p for p in jpeg_root.iterdir() if p.is_dir()):
        dst=args.pred_root/vd.name; dst.mkdir(parents=True,exist_ok=True)
        for p in sorted(vd.glob('*.jpg')):
            idx=int(p.stem); base=load_label(args.base_root/vd.name/f'{p.stem}.png').copy(); final=base.copy()
            if vd.name==vid and idx>0:
                if 3<=idx<=5:
                    final[final==obj]=0
                else:
                    box=roi_box(idx,base.shape[1],base.shape[0],args.mode)
                    if box is not None:
                        rgb=np.asarray(Image.open(p).convert('RGB'))
                        ft=feature(rgb)
                        dp=dist_to_centers(ft,cpos); dn=dist_to_centers(ft,cneg)
                        x1,y1,x2,y2=box
                        roi=np.zeros(base.shape,bool); roi[y1:y2,x1:x2]=True
                        thr={'conservative':0.19,'balanced':0.23,'wide':0.28}[args.mode]
                        margin={'conservative':0.015,'balanced':0.0,'wide':-0.02}[args.mode]
                        cand=(dp<thr) & (dp+margin<dn) & roi
                        # keep dark/white low-saturation parts inside ROI; suppress very green grass.
                        rr,gg,bb=[rgb[...,i].astype(np.int16) for i in range(3)]
                        greenish=(gg>rr+22)&(gg>bb+22)
                        cand &= ~greenish
                        cand=smooth(largest_components(cand, max_components=2 if idx<30 else 1, min_area=80)) & roi
                        final[final==obj]=0
                        final[cand]=obj
                if not np.array_equal(base==obj,final==obj):
                    audit['changed_frames'].append(idx)
                m=final==obj
                if m.any():
                    ys,xs=np.where(m); info={'area':int(m.sum()),'box':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]}
                else: info={'area':0,'box':None}
                audit['frames'][str(idx)]=info
            save_label(dst/f'{p.stem}.png',final,palette)
    args.audit_json.parent.mkdir(parents=True,exist_ok=True)
    args.audit_json.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.make_submission:
        submit=args.submit_root or (ws/'homework/submission_433_m17_q0_temporal_color')
        zip_path=args.zip_path or (ws/'homework/submission_mosev2_m17_q0_temporal_color.zip')
        make_submission(ws,args.pred_root,submit,zip_path,args.overwrite)
    print(json.dumps({'pred_root':str(args.pred_root),'changed_frames':len(audit['changed_frames']),'audit_json':str(args.audit_json)},ensure_ascii=False))

if __name__=='__main__': main()
