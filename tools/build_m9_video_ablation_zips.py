#!/usr/bin/env python3
"""Build one-change-at-a-time MOSEv2 submission zips for hidden-score ablation.

Each case starts from a default 15-video prediction root (usually M11) and
replaces exactly one reviewed video/object from a candidate root.  The script
then builds a full 433-video Codabench zip through ``apply_m6_safe_fusion.py``
and validates the 418-provided-output invariant.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
APPLY = REPO_ROOT / "tools" / "apply_m6_safe_fusion.py"
VALIDATE = REPO_ROOT / "tools" / "validate_mose_submission.py"


@dataclass(frozen=True)
class Case:
    name: str
    video: str
    obj_id: int | None
    root: Path
    note: str = ""

    @property
    def replace_rule(self) -> str:
        left = self.video if self.obj_id is None else f"{self.video}:{self.obj_id}"
        return f"{left}={self.root}"


def parse_case(text: str) -> Case:
    """Parse name=video[:obj]=/candidate/root[#note]."""
    note = ""
    if "#" in text:
        text, note = text.split("#", 1)
    if text.count("=") < 2:
        raise argparse.ArgumentTypeError("case must be name=video[:obj]=/candidate/root[#note]")
    name, rest = text.split("=", 1)
    target, root = rest.split("=", 1)
    if ":" in target:
        video, obj = target.split(":", 1)
        obj_id: int | None = int(obj)
    else:
        video = target
        obj_id = None
    if not name or not video or not root:
        raise argparse.ArgumentTypeError(f"bad case: {text}")
    return Case(name=name, video=video, obj_id=obj_id, root=Path(root).expanduser(), note=note)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--default-root", type=Path, required=True)
    p.add_argument("--case", action="append", type=parse_case, required=True, help="name=video[:obj]=/candidate/root[#note]")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: <workspace>/homework")
    p.add_argument("--audit-dir", type=Path, default=None, help="Default: <workspace>/homework/logs/m9_ablation")
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout, proc.stderr


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    output_dir = (args.output_dir or ws / "homework").resolve()
    audit_dir = (args.audit_dir or ws / "homework" / "logs" / "m9_ablation").resolve()
    summary_json = (args.summary_json or audit_dir / "m9_video_ablation_summary.json").resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for case in args.case:
        pred_root = output_dir / f"pred_m9_ablation_{case.name}"
        submit_root = output_dir / f"submit_m9_ablation_{case.name}"
        zip_path = output_dir / f"submission_mosev2_m9_ablation_{case.name}.zip"
        audit_json = audit_dir / f"{case.name}.json"
        validation_json = audit_dir / f"{case.name}_validation.json"
        cmd = [
            sys.executable,
            str(APPLY),
            "--workspace",
            str(ws),
            "--default-root",
            str(args.default_root.resolve()),
            "--pred-root",
            str(pred_root),
            "--submit-root",
            str(submit_root),
            "--zip-path",
            str(zip_path),
            "--audit-json",
            str(audit_json),
            "--replace",
            case.replace_rule,
            "--note",
            f"M9 single-object hidden-score ablation: {case.note or case.replace_rule}",
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        rc, out, err = run(cmd)
        rec: dict[str, Any] = {
            "case": asdict(case) | {"root": str(case.root)},
            "replace_rule": case.replace_rule,
            "pred_root": str(pred_root),
            "submit_root": str(submit_root),
            "zip_path": str(zip_path),
            "audit_json": str(audit_json),
            "apply_returncode": rc,
            "apply_stdout": out[-4000:],
            "apply_stderr": err[-4000:],
        }
        if rc == 0:
            vcmd = [
                sys.executable,
                str(VALIDATE),
                "--workspace",
                str(ws),
                "--zip-path",
                str(zip_path),
                "--output-json",
                str(validation_json),
            ]
            vrc, vout, verr = run(vcmd)
            rec.update(
                {
                    "validation_json": str(validation_json),
                    "validation_returncode": vrc,
                    "validation_stdout": vout[-4000:],
                    "validation_stderr": verr[-4000:],
                }
            )
            if validation_json.is_file():
                try:
                    rec["validation"] = json.loads(validation_json.read_text(encoding="utf-8"))
                except Exception:
                    pass
        records.append(rec)
        print(json.dumps({"case": case.name, "zip_path": str(zip_path), "apply_ok": rc == 0, "validation_ok": rec.get("validation", {}).get("ok")}, ensure_ascii=False), flush=True)
    summary = {"workspace": str(ws), "default_root": str(args.default_root.resolve()), "records": records}
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_json), "cases": len(records)}, ensure_ascii=False, indent=2))
    failed = [r for r in records if r.get("apply_returncode") != 0 or not r.get("validation", {}).get("ok", False)]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
