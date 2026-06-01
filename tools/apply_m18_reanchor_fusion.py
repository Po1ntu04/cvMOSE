#!/usr/bin/env python3
"""Apply M18 review-gated object/frame fusion.

Unlike earlier probe fusion scripts, this tool refuses to copy a candidate into a
submission tier unless the edit has an explicit promotion/review decision and is
compatible with the object ledger.  It still performs simple object-id mask
replacement; SAM2/SAM3 propagation must happen upstream.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.temporal_ledger import ledger_key, load_ledgers  # noqa: E402

MODE_ORDER = {"safe": 0, "balanced": 1, "aggressive": 2}
HIGH_RISK_TAGS = {
    "same_class_dense_unreviewed",
    "hard_negative_similarity",
    "semantic_swap",
    "aggressive_regression",
    "old_position_distractor",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--candidate-roots-json", type=Path, required=True, help="{name:path}; source may also be 'empty'")
    p.add_argument("--policy-json", type=Path, required=True, help="{'edits':[...]} or list of edits")
    p.add_argument("--ledger-dir", type=Path, default=Path("artifacts/m18_temporal_ledger"))
    p.add_argument("--mode", choices=["safe", "balanced", "aggressive"], required=True)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--source-table-csv", type=Path, default=None)
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr


def save_label(path: Path, arr: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr.astype(np.uint8), mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def frame_indices(spec: dict[str, Any], n: int) -> set[int]:
    if "frames" in spec:
        return {int(x) for x in spec["frames"] if 0 <= int(x) < n}
    start = int(spec.get("start", 1))
    end = int(spec.get("end", n - 1))
    return set(range(max(1, start), min(n - 1, end) + 1))


def tier_allowed(edit_tier: str, mode: str) -> bool:
    if edit_tier not in MODE_ORDER:
        edit_tier = "aggressive"
    return MODE_ORDER[edit_tier] <= MODE_ORDER[mode]


def gate_edit(edit: dict[str, Any], ledger: dict[str, Any] | None, mode: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not tier_allowed(str(edit.get("tier", "aggressive")), mode):
        reasons.append("tier_not_allowed_for_mode")
    decision = str(edit.get("decision", edit.get("promotion_decision", "needs_more_evidence")))
    review_status = str(edit.get("review_status", "unreviewed"))
    risk_tags = {str(x) for x in edit.get("risk_tags", [])}
    source = str(edit.get("source", ""))
    is_empty_source = source == "empty"
    if decision == "reject" or review_status == "rejected":
        reasons.append("edit_rejected")
    if ledger is None:
        reasons.append("missing_temporal_ledger")
    else:
        if mode == "safe" and ledger.get("review_status") != "approved":
            reasons.append("safe_requires_approved_ledger")
        if any(tag in HIGH_RISK_TAGS for tag in risk_tags) and review_status != "approved":
            reasons.append("high_risk_requires_review_approval")
        same_class_dense = bool((ledger.get("target_profile") or {}).get("same_class_dense"))
        if (
            mode in {"safe", "balanced"}
            and same_class_dense
            and not is_empty_source
            and review_status != "approved"
            and not bool(edit.get("balanced_allowed"))
        ):
            reasons.append("same_class_dense_requires_approved_or_balanced_allowed_edit")
    if mode == "safe":
        if decision != "promote":
            reasons.append("safe_requires_promote_decision")
        if review_status != "approved":
            reasons.append("safe_requires_approved_edit")
        if risk_tags.intersection(HIGH_RISK_TAGS):
            reasons.append("safe_blocks_high_risk_tags")
    elif mode == "balanced":
        if decision not in {"promote", "output_only"}:
            reasons.append("balanced_requires_promote_or_output_only")
        if review_status not in {"approved", "needs_more_evidence"} and not bool(edit.get("balanced_allowed")):
            reasons.append("balanced_requires_review_material")
    else:  # aggressive
        if decision not in {"promote", "output_only", "needs_more_evidence"}:
            reasons.append("aggressive_blocks_unknown_decision")
    return not reasons, reasons


def make_submission(workspace: Path, pred_root: Path, submit_root: Path, zip_path: Path, overwrite: bool) -> dict[str, Any]:
    from validate_mose_submission import validate_zip

    provided = workspace / "homework" / "output"
    jpeg_root = workspace / "homework" / "JPEGImages"
    ann_root = workspace / "homework" / "Annotations"
    if submit_root.exists() and overwrite:
        shutil.rmtree(submit_root)
    submit_root.mkdir(parents=True, exist_ok=True)
    for root in [provided, pred_root]:
        for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            dst = submit_root / video_dir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(video_dir, dst)
    if zip_path.exists() and overwrite:
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for png in sorted(submit_root.rglob("*.png")):
            zf.write(png, png.relative_to(submit_root).as_posix())
    result = validate_zip(zip_path, provided, jpeg_root, ann_root, 433, 66526, 15, False)
    if not result.get("ok"):
        raise SystemExit("submission validation failed: " + json.dumps(result, ensure_ascii=False))
    return result


def root_name_from_source(source: str) -> str:
    return source.split(":", 1)[0]


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = ws / "homework" / "JPEGImages"
    ann_root = ws / "homework" / "Annotations"
    ledgers = load_ledgers(args.ledger_dir)
    roots = {k: Path(v).expanduser().resolve() for k, v in json.loads(args.candidate_roots_json.read_text(encoding="utf-8")).items()}
    raw_policy = json.loads(args.policy_json.read_text(encoding="utf-8"))
    edits = raw_policy.get("edits", raw_policy if isinstance(raw_policy, list) else [])
    allowed_edits: list[dict[str, Any]] = []
    rejected_edits: list[dict[str, Any]] = []
    for edit in edits:
        key = ledger_key(str(edit["video"]), int(edit["obj_id"]))
        ok, reasons = gate_edit(edit, ledgers.get(key), args.mode)
        record = dict(edit)
        record["ledger_key"] = key
        record["gate_reasons"] = reasons
        if ok:
            allowed_edits.append(record)
        else:
            rejected_edits.append(record)
    by_video: dict[str, list[dict[str, Any]]] = {}
    for edit in allowed_edits:
        by_video.setdefault(str(edit["video"]), []).append(edit)

    if args.pred_root.exists() and args.overwrite:
        shutil.rmtree(args.pred_root)
    args.pred_root.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "method": "m18_reanchor_fusion",
        "mode": args.mode,
        "base_root": str(args.base_root),
        "candidate_roots": {k: str(v) for k, v in roots.items()},
        "policy_json": str(args.policy_json),
        "allowed_edit_count": len(allowed_edits),
        "rejected_edit_count": len(rejected_edits),
        "allowed_edits": allowed_edits,
        "rejected_edits": rejected_edits,
        "videos": {},
    }
    for video_dir in sorted(p for p in jpeg_root.iterdir() if p.is_dir()):
        video = video_dir.name
        frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
        if not frames:
            continue
        dst = args.pred_root / video
        dst.mkdir(parents=True, exist_ok=True)
        ann_img = Image.open(ann_root / video / "00000.png")
        palette = ann_img.getpalette()
        video_edits = by_video.get(video, [])
        frame_edit_map: dict[int, list[dict[str, Any]]] = {}
        for edit in video_edits:
            for idx in frame_indices(edit, len(frames)):
                frame_edit_map.setdefault(idx, []).append(edit)
        changed_frames: set[int] = set()
        for idx, frame in enumerate(frames):
            base = load_label(args.base_root / video / f"{frame.stem}.png").copy()
            final = base.copy()
            if idx == 0:
                final = load_label(ann_root / video / "00000.png").copy()
            else:
                for edit in frame_edit_map.get(idx, []):
                    obj_id = int(edit["obj_id"])
                    source = str(edit["source"])
                    final[final == obj_id] = 0
                    if source == "empty":
                        source_area = 0
                    else:
                        root_name = root_name_from_source(source)
                        if root_name not in roots:
                            raise SystemExit(f"unknown candidate root {root_name!r} for edit {edit}")
                        cand = load_label(roots[root_name] / video / f"{frame.stem}.png")
                        cm = cand == obj_id
                        final[cm] = obj_id
                        source_area = int(cm.sum())
                    source_rows.append(
                        {
                            "video": video,
                            "obj_id": obj_id,
                            "frame_idx": idx,
                            "source": source,
                            "tier": edit.get("tier", "aggressive"),
                            "mode": args.mode,
                            "source_area": source_area,
                            "reason": edit.get("reason", edit.get("decision_reason", "")),
                        }
                    )
            if not np.array_equal(base, final):
                changed_frames.add(idx)
            save_label(dst / f"{frame.stem}.png", final, palette)
        audit["videos"][video] = {
            "changed_frames": sorted(changed_frames),
            "changed_frame_count": len(changed_frames),
            "edit_count": len(video_edits),
        }
    audit["source_table"] = source_rows
    audit["summary"] = {
        "videos": len(audit["videos"]),
        "changed_videos": sum(1 for v in audit["videos"].values() if v["changed_frame_count"]),
        "changed_frame_events": len(source_rows),
    }
    if args.make_submission:
        submit_root = args.submit_root or (ws / "homework" / f"submission_433_m18_{args.mode}")
        zip_path = args.zip_path or (ws / "homework" / f"submission_mosev2_m18_{args.mode}.zip")
        audit["validation"] = make_submission(ws, args.pred_root, submit_root, zip_path, args.overwrite)
        audit["zip_path"] = str(zip_path)
        audit["submit_root"] = str(submit_root)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.source_table_csv or args.audit_json.with_suffix(".source_table.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["video", "obj_id", "frame_idx", "source", "tier", "mode", "source_area", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)
    print(json.dumps({"pred_root": str(args.pred_root), "audit_json": str(args.audit_json), "source_table_csv": str(csv_path), "summary": audit["summary"], "allowed_edit_count": len(allowed_edits), "rejected_edit_count": len(rejected_edits)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
