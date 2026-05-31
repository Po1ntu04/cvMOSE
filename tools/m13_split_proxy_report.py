#!/usr/bin/env python3
"""Build a local proxy report for split-harness event stories.

This tool is intentionally lightweight: it does not score against hidden GT.  It
summarizes MLLM event/action signals plus disagreement among available prediction
roots so each optimization round can decide which objects are worth probing.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.mllm_panels import first_annotation, homework_roots, list_frames  # noqa: E402

DEFAULT_ROOTS = {
    "m7": "pred_m7_part_balanced",
    "sam2": "pred_sam2_b101",
    "m11": "pred_sam2_m11_cycle",
    "official_l": "pred_sam2_official_large_mosev2",
    "official_b": "pred_sam2_official_bplus_mosev2",
    "m7_tiny": "pred_sam2_m7_qwen_support_tiny_semantic",
    "m10_q0": "pred_m10_qwen36_q0only_fusion",
    "m10_q04v": "pred_m10_qwen36_q0_4v_fusion",
}

P0_HINT = {
    ("8jsm23a7", 1),
    ("q0sizv6m", 2),
    ("z6dx46qr", 1),
    ("amfdu83t", 1),
}

HIGH_LEVERAGE = {"8jsm23a7", "q0sizv6m", "z6dx46qr", "amfdu83t", "r13u5z4y", "1qlssuz2", "4vznweiu"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-doc", type=Path, required=True)
    p.add_argument("--roots-json", type=Path, default=None)
    return p.parse_args()


def load_label(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr


def iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    aa = a.astype(bool); bb = b.astype(bool)
    union = int(np.logical_or(aa, bb).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(aa, bb).sum() / union)


def mask_stats(workspace: Path, roots: dict[str, str], video: str, obj_id: int) -> dict[str, Any]:
    jpeg_root, ann_root = homework_roots(workspace)
    frames = list_frames(jpeg_root, video)
    _, ann = first_annotation(ann_root, video)
    init_area = int((ann == int(obj_id)).sum())
    out: dict[str, Any] = {"frames": len(frames), "init_area": init_area, "roots": {}}
    masks_by_root: dict[str, list[np.ndarray | None]] = {}
    for name, root in roots.items():
        root_dir = workspace / "homework" / root / video
        areas: list[int] = []
        masks: list[np.ndarray | None] = []
        missing = 0
        for frame in frames:
            arr = load_label(root_dir / f"{frame.stem}.png")
            if arr is None:
                missing += 1
                masks.append(None)
                areas.append(-1)
            else:
                m = arr == int(obj_id)
                masks.append(m)
                areas.append(int(m.sum()))
        masks_by_root[name] = masks
        valid = [x for x in areas if x >= 0]
        out["roots"][name] = {
            "missing": missing,
            "empty_count": sum(1 for x in valid if x == 0),
            "nonempty_count": sum(1 for x in valid if x > 0),
            "mean_area": round(float(np.mean(valid)) if valid else 0.0, 2),
            "max_area": int(max(valid) if valid else 0),
        }
    if "m7" in masks_by_root:
        for name, masks in masks_by_root.items():
            if name == "m7":
                continue
            vals = [iou(a, b) for a, b in zip(masks_by_root["m7"], masks)]
            vals2 = [v for v in vals if v is not None]
            out["roots"].setdefault(name, {})["mean_iou_vs_m7"] = round(float(np.mean(vals2)) if vals2 else 0.0, 4)
            out["roots"].setdefault(name, {})["low_iou_frames_vs_m7"] = sum(1 for v in vals2 if v < 0.5)
    return out


def action_priority(rec: dict[str, Any], stats: dict[str, Any]) -> tuple[str, int, list[str]]:
    video = rec["video"]; obj_id = int(rec["obj_id"])
    j = rec.get("judgment", {}) or {}
    event = j.get("event_type")
    diagnosis = j.get("current_prediction_diagnosis")
    action = j.get("recommended_action")
    conf = float(j.get("confidence") or 0.0)
    reasons: list[str] = []
    score = 0
    if (video, obj_id) in P0_HINT:
        score += 5; reasons.append("manual_P0")
    if video in HIGH_LEVERAGE:
        score += 2; reasons.append("high_leverage_video")
    if event in {"picked_moved_placed", "moves_to_foreground", "exits_reappears", "low_contrast_continuation"}:
        score += 2; reasons.append(f"event:{event}")
    if diagnosis in {"wrong_static_distractor", "wrong_same_class", "empty_when_visible", "uncertain"}:
        score += 2; reasons.append(f"diagnosis:{diagnosis}")
    if action in {"reanchor_at_frame", "use_official_interval", "generate_detector_candidates"}:
        score += 2; reasons.append(f"action:{action}")
    if conf >= 0.80:
        score += 1; reasons.append("high_mllm_conf")
    roots = stats.get("roots", {})
    for name in ["official_l", "official_b", "m10_q0", "m10_q04v", "m7_tiny"]:
        low = roots.get(name, {}).get("low_iou_frames_vs_m7", 0)
        if low and low >= max(2, stats.get("frames", 0) // 8):
            score += 1; reasons.append(f"source_disagrees:{name}")
            break
    if score >= 9:
        return "P0", score, reasons
    if score >= 6:
        return "P1", score, reasons
    if score >= 3:
        return "P2", score, reasons
    return "drop", score, reasons


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# M13 Round 1 split-harness proxy report",
        "",
        f"- split_json: `{payload['split_json']}`",
        f"- records: `{len(rows)}`",
        f"- aggregate_success_rate: `{payload['aggregate_success_rate']:.2%}`",
        "",
        "## Ranked action table",
        "",
        "| priority | target | event | diagnosis | action | conf | proxy | key reasons |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['priority']} | `{row['video']}:{row['obj_id']}` | {row['event_type']} | {row['diagnosis']} | {row['action']} | {row['confidence']:.2f} | {row['proxy_score']} | {', '.join(row['priority_reasons'][:4])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- P0 means event-chain intervention is likely worth a bounded SAM/proxy experiment.",
        "- This report is not hidden-GT evaluation; it is a local routing/audit layer for deciding which probes deserve zip candidates.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    roots = dict(DEFAULT_ROOTS)
    if args.roots_json and args.roots_json.is_file():
        roots.update(json.loads(args.roots_json.read_text(encoding="utf-8")))
    data = json.loads(args.split_json.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    ok = 0
    for rec in data.get("records", []):
        j = rec.get("judgment", {}) or {}
        if j.get("status") == "ok":
            ok += 1
        stats = mask_stats(args.workspace, roots, str(rec["video"]), int(rec["obj_id"]))
        priority, score, reasons = action_priority(rec, stats)
        row = {
            "video": str(rec["video"]),
            "obj_id": int(rec["obj_id"]),
            "status": j.get("status"),
            "event_type": j.get("event_type"),
            "diagnosis": j.get("current_prediction_diagnosis"),
            "action": j.get("recommended_action"),
            "confidence": float(j.get("confidence") or 0.0),
            "priority": priority,
            "proxy_score": int(score),
            "priority_reasons": reasons,
            "frame_status_counts": rec.get("frame_status_counts", {}),
            "positive_prompt_plan": j.get("positive_prompt_plan", []),
            "hard_negatives": j.get("hard_negatives", []),
            "sam_teaching_plan": j.get("sam_teaching_plan", {}),
            "mask_stats": stats,
        }
        rows.append(row)
    rows.sort(key=lambda r: ({"P0": 0, "P1": 1, "P2": 2, "drop": 3}.get(r["priority"], 9), -r["proxy_score"], r["video"], r["obj_id"]))
    payload = {
        "split_json": str(args.split_json),
        "aggregate_success_rate": ok / max(1, len(data.get("records", []))),
        "roots": roots,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["priority", "video", "obj_id", "status", "event_type", "diagnosis", "action", "confidence", "proxy_score", "priority_reasons"])
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in writer.fieldnames if k in row}
            flat["priority_reasons"] = ";".join(row["priority_reasons"])
            writer.writerow(flat)
    write_doc(args.out_doc, payload)
    print(json.dumps({"rows": len(rows), "success_rate": payload["aggregate_success_rate"], "out_json": str(args.out_json), "out_doc": str(args.out_doc)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
