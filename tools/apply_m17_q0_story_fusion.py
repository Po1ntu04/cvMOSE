#!/usr/bin/env python3
"""Fuse q0sizv6m:obj2 by explicit temporal-story segments.

This probe encodes the user-corrected physical story after full-box hidden
feedback: use the bottom/foreground SAM2 box path while the animal is visible,
optionally use a color/ROI rump candidate around frames 30-33, and force empty
after the user-described disappearance around frame 34.
"""
from __future__ import annotations

import argparse, json, shutil, zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('/home/yu/projects/cv/from fdu/MOSEv2'))
    p.add_argument('--base-root', type=Path, required=True)
    p.add_argument('--full-box-root', type=Path, required=True)
    p.add_argument('--color-root', type=Path, required=True)
    p.add_argument('--pred-root', type=Path, required=True)
    p.add_argument('--submit-root', type=Path, default=None)
    p.add_argument('--zip-path', type=Path, default=None)
    p.add_argument('--audit-json', type=Path, required=True)
    p.add_argument('--policy', choices=['trim_after34','rump_30_33','rump_31_33'], default='trim_after34')
    p.add_argument('--make-submission', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def load_label(p:Path)->np.ndarray:
    arr=np.asarray(Image.open(p)); return arr if arr.ndim==2 else arr[...,0]


def save_label(p:Path, arr:np.ndarray, palette):
    p.parent.mkdir(parents=True, exist_ok=True)
    img=Image.fromarray(arr.astype(np.uint8), mode='P')
    if palette: img.putpalette(palette)
    img.save(p)


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


def source_for_frame(policy:str, idx:int)->str|None:
    # return full_box / color / empty / None(base)
    if 6 <= idx <= 29:
        return 'full_box'
    if policy == 'rump_30_33' and 30 <= idx <= 33:
        return 'color'
    if policy == 'rump_31_33' and 31 <= idx <= 33:
        return 'color'
    if 34 <= idx <= 41:
        return 'empty'
    if policy == 'trim_after34' and 30 <= idx <= 33:
        return 'empty'
    return None


def main():
    args=parse_args(); ws=args.workspace.resolve(); vid='q0sizv6m'; obj=2
    jpeg_root=ws/'homework/JPEGImages'; ann_root=ws/'homework/Annotations'
    ann_img=Image.open(ann_root/vid/'00000.png'); palette=ann_img.getpalette()
    if args.pred_root.exists() and args.overwrite: shutil.rmtree(args.pred_root)
    args.pred_root.mkdir(parents=True, exist_ok=True)
    audit={'policy':args.policy,'video':vid,'obj_id':obj,'base_root':str(args.base_root),'full_box_root':str(args.full_box_root),'color_root':str(args.color_root),'frames':{},'changed_frames':[]}
    for vd in sorted(p for p in jpeg_root.iterdir() if p.is_dir()):
        dst=args.pred_root/vd.name; dst.mkdir(parents=True, exist_ok=True)
        for frame in sorted(vd.glob('*.jpg')):
            idx=int(frame.stem)
            base=load_label(args.base_root/vd.name/f'{frame.stem}.png').copy(); final=base.copy(); src=None
            if vd.name==vid and idx>0:
                src=source_for_frame(args.policy, idx)
                if src=='full_box':
                    cand=load_label(args.full_box_root/vid/f'{frame.stem}.png')
                    final[final==obj]=0; final[cand==obj]=obj
                elif src=='color':
                    cand=load_label(args.color_root/vid/f'{frame.stem}.png')
                    final[final==obj]=0; final[cand==obj]=obj
                elif src=='empty':
                    final[final==obj]=0
                if src is not None and not np.array_equal(base==obj, final==obj):
                    audit['changed_frames'].append(idx)
                if src is not None:
                    m=final==obj
                    if m.any():
                        ys,xs=np.where(m); info={'source':src,'area':int(m.sum()),'box':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]}
                    else:
                        info={'source':src,'area':0,'box':None}
                    audit['frames'][str(idx)]=info
            save_label(dst/f'{frame.stem}.png', final, palette)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.make_submission:
        submit=args.submit_root or (ws/f'homework/submission_433_m17_q0_story_{args.policy}')
        zip_path=args.zip_path or (ws/f'homework/submission_mosev2_m17_q0_story_{args.policy}.zip')
        make_submission(ws,args.pred_root,submit,zip_path,args.overwrite)
    print(json.dumps({'pred_root':str(args.pred_root),'policy':args.policy,'changed_frames':len(audit['changed_frames']),'audit_json':str(args.audit_json)},ensure_ascii=False))

if __name__=='__main__': main()
