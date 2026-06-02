#!/usr/bin/env python3
"""Build M8 high-recall candidate-pool and conservative source-selection audit.

This tool does **not** use hidden labels and does **not** generate masks.  It
answers the M8 first question: do any existing/proposal roots contain plausible
post-gap candidates, and which ones are safe enough to copy or pass to SAM2
re-anchor?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.candidate_pool import (  # noqa: E402
    CandidateRecord,
    connected_components,
    cosine,
    descriptor,
    label_path,
    list_frames,
    load_label,
    load_rgb,
    mask_iou,
    mask_stats,
    parse_source_roots,
    ring_mask,
)

SAME_CLASS_DENSE = {"r13u5z4y", "q0sizv6m", "msinig6m", "2smf7uq9", "8jsm23a7", "4vznweiu"}
STABLE_GUARD = {"8jsm23a7", "jadgtmfl"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--default-root", type=Path, required=True, help="Usually M11")
    p.add_argument("--source-root", action="append", default=[], help="name=/path; repeatable")
    p.add_argument("--judgments-json", action="append", type=Path, default=[])
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--min-area", type=int, default=8)
    p.add_argument("--component-min-area", type=int, default=12)
    p.add_argument("--component-max-count", type=int, default=10)
    p.add_argument("--top-k-per-object", type=int, default=30)
    p.add_argument("--accept-margin", type=float, default=0.08)
    p.add_argument("--accept-pos", type=float, default=0.50)
    p.add_argument("--sameclass-margin", type=float, default=0.18)
    p.add_argument("--max-area-ratio-init", type=float, default=8.0)
    p.add_argument("--min-area-ratio-init", type=float, default=0.015)
    p.add_argument("--allow-stable-guard", action="store_true")
    return p.parse_args()


def qwen_records(paths: list[Path]) -> tuple[dict[tuple[str, int, int, str], dict[str, Any]], dict[tuple[str, int, int], dict[str, Any]]]:
    out: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    frame_veto: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data.get("records", []):
            j = rec.get("judgment", rec)
            best = str(j.get("best_candidate", ""))
            src = None
            if len(best) == 1 and best.isalpha():
                for c in rec.get("candidates", []):
                    if str(c.get("candidate_id")) == best:
                        src = str(c.get("source")); break
            if not src and best == "keep_baseline":
                src = "baseline"
            try:
                frame_key = (str(rec["video"]), int(rec["obj_id"]), int(rec["frame_idx"]))
                key = (*frame_key, str(src or ""))
            except Exception:
                continue
            # Preserve explicit frame-level rejections such as EMPTY / reject_all;
            # otherwise the veto disappears because there is no candidate source
            # to match against.
            if (
                (
                    bool(j.get("should_veto_anchor"))
                    and str(j.get("best_candidate", "")) in {"empty", "none", "uncertain"}
                )
                or str(j.get("recommended_action", "")) in {"reject_all", "keep_empty"}
            ):
                frame_veto[frame_key] = rec
            if src:
                out[key] = rec
    return out, frame_veto


def source_match(candidate_source: str, qwen_source: str) -> bool:
    if not qwen_source:
        return False
    return candidate_source == qwen_source or candidate_source.split(":", 1)[0] == qwen_source.split(":", 1)[0]


def load_root_labels(root: Path, video: str, frames: list[Path], shape: tuple[int, int]) -> list[np.ndarray | None]:
    return [load_label(label_path(root, video, f.stem), shape) for f in frames]


def risk_frames(frames: list[Path], obj_id: int, labels: dict[str, list[np.ndarray | None]], default_name: str = "default") -> set[int]:
    out: set[int] = set()
    base = labels.get("baseline") or labels.get(default_name) or []
    default = labels.get(default_name) or []
    prev_area = None
    for i in range(1, len(frames)):
        b = base[i] == obj_id if i < len(base) and base[i] is not None else None
        d = default[i] == obj_id if i < len(default) and default[i] is not None else None
        barea = int(b.sum()) if b is not None else 0
        darea = int(d.sum()) if d is not None else barea
        if barea == 0 or darea == 0 or (barea > 0 and darea == 0):
            out.add(i)
        if prev_area and prev_area > 0:
            ratio = barea / max(prev_area, 1)
            if ratio < 0.25 or ratio > 4.0:
                out.add(i)
        areas = []
        for seq in labels.values():
            if i < len(seq) and seq[i] is not None:
                areas.append(int((seq[i] == obj_id).sum()))
        pos = [a for a in areas if a > 0]
        if pos and (min(pos) == 0 or (max(pos) / max(1, min(pos)) > 2.5)):
            out.add(i)
        if barea > 0:
            prev_area = barea
    padded = set(out)
    for i in list(out):
        for j in range(max(1, i - 2), min(len(frames), i + 3)):
            padded.add(j)
    return padded


def build_banks(video: str, frames: list[Path], ann: np.ndarray, obj_id: int, labels: dict[str, list[np.ndarray | None]], risks: set[int], args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    pos: list[np.ndarray] = []
    neg: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []

    def add(kind: str, frame_idx: int, source: str, mask: np.ndarray) -> None:
        rgb = load_rgb(frames[frame_idx])
        desc = descriptor(rgb, mask)
        if desc is None:
            return
        (pos if kind == "positive" else neg).append(desc)
        audit.append({"kind": kind, "frame_idx": frame_idx, "source": source, "area": int(mask.sum())})

    init = ann == obj_id
    add("positive", 0, "first_frame_gt", init)
    ring = ring_mask(init)
    if int(ring.sum()) >= args.component_min_area:
        add("negative", 0, "first_frame_context_ring", ring)
    baseline = labels.get("baseline") or labels.get("default") or []
    default = labels.get("default") or []
    for i in range(1, len(frames)):
        if len(pos) >= 8:
            break
        if i in risks:
            continue
        if i < len(baseline) and baseline[i] is not None:
            m = baseline[i] == obj_id
            if int(m.sum()) >= args.component_min_area and not (i < len(default) and default[i] is not None and int((default[i] == obj_id).sum()) == 0):
                add("positive", i, "baseline_stable", m)
    for i in range(1, len(frames)):
        if len(neg) >= 64:
            break
        if i < len(baseline) and baseline[i] is not None:
            bm = baseline[i] == obj_id
            if i < len(default) and default[i] is not None and int(bm.sum()) >= args.component_min_area and int((default[i] == obj_id).sum()) == 0:
                add("negative", i, "default_rejected_baseline", bm)
            for other in [int(x) for x in np.unique(baseline[i]) if int(x) not in {0, obj_id}]:
                om = baseline[i] == other
                if int(om.sum()) >= args.component_min_area:
                    add("negative", i, f"baseline_other:{other}", om)
        for name, seq in labels.items():
            if len(neg) >= 64 or i >= len(seq) or seq[i] is None:
                continue
            lab = seq[i]
            obj = lab == obj_id
            for k, comp in enumerate(connected_components(lab > 0, args.component_min_area, min(args.component_max_count, 8))):
                if mask_iou(comp, obj) < 0.10:
                    add("negative", i, f"{name}:anyfg_negative:{k}", comp)
                    if len(neg) >= 64:
                        break
    return pos, neg, audit


def add_qwen(c: CandidateRecord, qwen: dict[tuple[str, int, int, str], dict[str, Any]], frame_veto: dict[tuple[str, int, int], dict[str, Any]]) -> None:
    fv = frame_veto.get((c.video, int(c.obj_id), int(c.frame_idx)))
    if fv is not None:
        j = fv.get("judgment", fv)
        conf = float(j.get("confidence") or 0.0)
        if conf >= 0.55:
            c.qwen_confidence = conf
            c.qwen_reason = j.get("reason_short")
            c.qwen_panel = fv.get("panel_path")
            c.qwen_veto = True
            return
    for (v, o, f, src), rec in qwen.items():
        if v == c.video and o == c.obj_id and f == c.frame_idx and source_match(c.source, src):
            j = rec.get("judgment", rec)
            conf = float(j.get("confidence") or 0.0)
            c.qwen_confidence = conf
            c.qwen_reason = j.get("reason_short")
            c.qwen_panel = rec.get("panel_path")
            c.qwen_support = bool(j.get("should_support_anchor")) and conf >= 0.70
            c.qwen_veto = bool(j.get("should_veto_anchor")) and conf >= 0.55
            return


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = ws / "homework" / "JPEGImages"
    ann_root = ws / "homework" / "Annotations"
    default_root = args.default_root.resolve()
    roots = [s for s in parse_source_roots(args.source_root) if s.root.resolve() != default_root]
    roots.insert(0, parse_source_roots([f"default={default_root}"])[0])
    if not any(s.name == "baseline" for s in roots):
        b = ws / "homework" / "pred_sam2_b101"
        if b.is_dir():
            roots.append(parse_source_roots([f"baseline={b}"])[0])
    qwen, frame_veto = qwen_records(args.judgments_json)
    videos = args.videos or sorted(p.name for p in jpeg_root.iterdir() if p.is_dir())
    aggregate: dict[str, Any] = {"method": "m8_candidate_pool_audit", "workspace": str(ws), "default_root": str(default_root), "source_roots": {s.name: str(s.root) for s in roots}, "videos": {}}
    csv_rows: list[dict[str, Any]] = []
    for video in videos:
        frames = list_frames(jpeg_root / video)
        ann = load_label(ann_root / video / "00000.png")
        if ann is None:
            continue
        shape = ann.shape
        obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
        labels = {s.name: load_root_labels(s.root, video, frames, shape) for s in roots if (s.root / video).is_dir()}
        vout: dict[str, Any] = {"frames": len(frames), "objects": {}}
        for obj_id in obj_ids:
            init_area = int((ann == obj_id).sum())
            risks = risk_frames(frames, obj_id, labels)
            pos, neg, bank = build_banks(video, frames, ann, obj_id, labels, risks, args)
            candidates: list[CandidateRecord] = []
            for idx in sorted(risks):
                rgb = load_rgb(frames[idx])
                seen: list[np.ndarray] = []
                default_label = labels.get("default", [None] * len(frames))[idx] if "default" in labels else None
                default_mask = default_label == obj_id if default_label is not None else None
                default_area = int(default_mask.sum()) if default_mask is not None else 0
                for name, seq in labels.items():
                    if idx >= len(seq) or seq[idx] is None:
                        continue
                    lab = seq[idx]
                    masks: list[tuple[str, np.ndarray]] = [(name, lab == obj_id)]
                    # High recall: any foreground components from each candidate root can expose wrong-label/composite proposals.
                    for k, comp in enumerate(connected_components(lab > 0, args.component_min_area, args.component_max_count)):
                        masks.append((f"{name}:anyfg:{k}", comp))
                    for src, mask in masks:
                        area = int(mask.sum())
                        if area < args.min_area:
                            continue
                        if any(mask_iou(mask, old) > 0.985 for old in seen):
                            continue
                        seen.append(mask.copy())
                        desc = descriptor(rgb, mask)
                        if desc is None:
                            continue
                        st = mask_stats(mask)
                        rec = CandidateRecord(
                            video=video,
                            obj_id=obj_id,
                            frame_idx=idx,
                            source=src,
                            root_name=name,
                            area=area,
                            bbox=st.bbox,
                            centroid=st.centroid,
                            area_ratio_init=area / max(1.0, float(init_area)),
                            area_ratio_default=None if default_area <= 0 else area / max(1.0, float(default_area)),
                            iou_default=None if default_mask is None else mask_iou(mask, default_mask),
                            pos_sim=max((cosine(desc, p) for p in pos), default=0.0),
                            neg_sim=max((cosine(desc, n) for n in neg), default=0.0),
                        )
                        rec.margin = rec.pos_sim - rec.neg_sim
                        # Cheap temporal support: same root has nearby non-empty similar object mask.
                        for j in [idx - 1, idx + 1, idx + 2]:
                            if 0 <= j < len(frames) and name in labels and labels[name][j] is not None:
                                nm = labels[name][j] == obj_id
                                if int(nm.sum()) >= args.min_area:
                                    rec.temporal_support += 1
                        add_qwen(rec, qwen, frame_veto)
                        rec.score = rec.margin + 0.015 * min(12.0, math.log1p(area)) + 0.03 * rec.temporal_support + (0.06 if rec.qwen_support else 0.0) - (0.20 if rec.qwen_veto else 0.0)
                        if rec.area_ratio_init > args.max_area_ratio_init:
                            rec.rejected.append("area_too_large_vs_init")
                        if rec.area_ratio_init < args.min_area_ratio_init:
                            rec.rejected.append("area_too_small_vs_init")
                        if rec.pos_sim < args.accept_pos:
                            rec.rejected.append("low_pos_similarity")
                        needed_margin = args.sameclass_margin if video in SAME_CLASS_DENSE else args.accept_margin
                        if rec.margin < needed_margin and not rec.qwen_support:
                            rec.rejected.append("low_identity_margin")
                        if rec.qwen_veto:
                            rec.rejected.append("qwen_veto")
                        if video in STABLE_GUARD and not args.allow_stable_guard:
                            rec.rejected.append("stable_guard")
                        if name == "default" or src.startswith("default"):
                            rec.rejected.append("same_as_default")
                        # Accept non-empty-to-non-empty root replacement only with very high confidence.
                        if default_area > 0 and area > 0 and (rec.area_ratio_default is not None) and not (0.75 <= rec.area_ratio_default <= 1.40):
                            rec.rejected.append("nonempty_area_ratio_guard")
                        candidates.append(rec)
            candidates.sort(key=lambda c: (len(c.rejected) == 0, c.score, c.margin, c.pos_sim), reverse=True)
            accepted = [c for c in candidates if not c.rejected][: args.top_k_per_object]
            top = candidates[: args.top_k_per_object]
            vout["objects"][str(obj_id)] = {
                "init_area": init_area,
                "risk_frames": sorted(int(x) for x in risks),
                "positive_bank": len(pos),
                "negative_bank": len(neg),
                "bank_items": bank[:80],
                "candidate_count": len(candidates),
                "accepted_count": len(accepted),
                "accepted": [c.to_dict() for c in accepted],
                "top_candidates": [c.to_dict() for c in top],
            }
            for c in top:
                row = c.to_dict(); csv_rows.append(row)
        aggregate["videos"][video] = vout
    aggregate["summary"] = {"videos": len(aggregate["videos"]), "objects": sum(len(v["objects"]) for v in aggregate["videos"].values()), "csv_rows": len(csv_rows), "accepted": sum(len(o["accepted"]) for v in aggregate["videos"].values() for o in v["objects"].values())}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video", "obj_id", "frame_idx", "source", "root_name", "area", "area_ratio_init", "area_ratio_default", "iou_default", "pos_sim", "neg_sim", "margin", "temporal_support", "score", "qwen_support", "qwen_veto", "qwen_confidence", "rejected"]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for row in csv_rows:
            w.writerow({k: row.get(k) for k in fieldnames})
    print(json.dumps({"out_json": str(args.out_json), "out_csv": str(args.out_csv), "summary": aggregate["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
